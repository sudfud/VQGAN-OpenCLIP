import modules
import utils

import argparse

import imageio

import numpy as np

import open_clip

import os

from pathlib import Path

from PIL import Image

import torch
from torch import optim
from torch.nn import functional as F

from torchvision import transforms
from torchvision.transforms import functional as TF

from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog = "VQGAN+OpenCLIP",
    description = "OpenCLIP-guided image generator.",
    epilog = "Have fun! :)"
)

parser.add_argument("prompts", help = "One or more weighted text prompts")
parser.add_argument("-m", "--model", help = "Generator model", default = "imagenet16384")
parser.add_argument("-s", "--seed", help = "Seed for RNG. -1 for random seed", default = -1)
parser.add_argument("--width", help = "Generated image width", default = 256)
parser.add_argument("--height", help = "Generated image height", default = 256)
parser.add_argument("--img_interval", help = "Training progress image interval", default = 50)
parser.add_argument("--init_image", help = "Initial image", default = "")
parser.add_argument("--target_imgs", help = "Image prompts", default = "")
parser.add_argument("--steps", help = "Number of training steps", default = 50)
parser.add_argument("--save_z", help = "Save latent vector to 'latents\\[name].pt'")

model_names = {
    "imagenet16384": "ImageNet 16384",
    "imagenet1024": "ImageNet 1024",
    "openimages16384": "OpenImages 16384",
    "openimages1024": "OpenImages 1024"
}

args = parser.parse_args()

prompts = args.prompts
seed = int(args.seed)
image_width = int(args.width)
image_height = int(args.height)
image_interval = int(args.img_interval)
init_image = args.init_image
target_images = args.target_imgs
steps = int(args.steps)
save_z = args.save_z is not None

model_name = model_names[args.model]
vqgan_config = f"vqgan_models\\{args.model}\\model.yaml"
vqgan_checkpoint = f"vqgan_models\\{args.model}\\last.ckpt"

init_weight = 0.0
clip_model_name = "ViT-B-32"
clip_pretrained = "laion2b_s34b_b79k"

step_size = 0.1
num_cutouts = 64
cutout_power = 1.0

z_file = ""

if init_image == "":
    init_image = None
elif init_image and init_image.lower().startswith("http"):
    init_image = utils.download_img(init_image)

if target_images == "" or not target_images:
    target_images = []
else:
    target_images = target_images.split('|')
    target_images = [image.strip() for image in target_images]

has_input_images = init_image or target_images != []

prompts = [prompt.strip() for prompt in prompts.split('|')]

if prompts == ['']:
    prompts = []

if save_z:
    z_file = args.save_z

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if prompts:
    print(f"Using prompts: {prompts}")

if target_images:
    print(f"Using image prompts: {target_images}")

if seed == -1:
    seed = torch.seed()

torch.manual_seed(seed)
print(f"Using seed: {seed}")

model = modules.load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)

noise_prompt_seeds = []
noise_prompt_weights = []

image_size = size = [image_width, image_height]

clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, clip_pretrained, jit = False, device = device)
clip_model.eval()
tokenizer = open_clip.get_tokenizer(clip_model_name)

cut_size = clip_model.visual.image_size[0]

embed_dims = model.quantize.e_dim

f = 2 ** (model.decoder.num_resolutions - 1)

make_cutouts = modules.MakeCutouts(cut_size, num_cutouts, cutout_power)

n_tokens = model.quantize.n_e

tokensX, tokensY = image_size[0] // f, image_size[1] // f
sideX, sideY = tokensX * f, tokensY * f

z_min = model.quantize.embedding.weight.min(dim = 0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim = 0).values[None, :, None, None]

if init_image:
    image = Image.open(init_image).convert("RGB")
    image = image.resize((sideX, sideY), Image.LANCZOS)

    z, *_ = model.encode(TF.to_tensor(image).to(device).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(torch.randint(n_tokens, [tokensX, tokensY], device = device), n_tokens).float()
    z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, tokensY, tokensX, embed_dims]).permute(0, 3, 1, 2)

z_orig = z.clone()
z.requires_grad_(True)

optimizer = optim.Adam([z], lr = step_size)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

pMs = []

for prompt in prompts:
    text, weight, stop = modules.parse_prompt(prompt)
    embed = clip_model.encode_text(tokenizer(text).to(device)).float()

    pMs.append(modules.Prompt(embed, weight, stop))

for prompt in target_images:
    path, weight, stop = modules.parse_prompt(prompt)
    image = modules.resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
    batch = make_cutouts(TF.pil_to_tensor(image).unsqueeze(0).to(device))
    embed = clip_model.encode_image(batch, normalize = True).float()

    pMs.append(modules.Prompt(embed, weight).to(device))

for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, clip_model.visual.output_dim]).normal_(generator = gen)
    pMs.append(modules.Prompt(embed, weight).to(device))

def synth(z):
    z_q = modules.vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)

    return modules.ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)

@torch.no_grad()
def check_in(i, losses):
    losses_str = ", ".join(f"{loss.item():g}" for loss in losses)

    tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")

    out = synth(z)

    if not os.path.exists(".\\progress"):
        Path.mkdir(".\\progress")

    TF.to_pil_image(out[0].cpu()).save(f"progress\\progress_{i:04}.png")

def ascend_text():
    out = synth(z)
    iii = clip_model.encode_image(make_cutouts(out), normalize = True).float()

    result = []

    if init_weight:
        result.append(F.mse_loss(z, z_orig) * init_weight / 2)

    for prompt in pMs:
        result.append(prompt(iii))

    image = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
    image = np.transpose(image, (1, 2, 0))

    if not os.path.exists(".\\steps"):
        Path.mkdir(".\\steps")
    filename = f"steps\\{i:04}.png"

    imageio.imwrite(filename, np.array(image))

    return result

def train(i):
    optimizer.zero_grad()

    loss_all = ascend_text()

    if i % image_interval == 0:
        check_in(i, loss_all)

    loss = sum(loss_all)
    loss.backward(retain_graph = True)
    
    optimizer.step()

    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))

try:
    with tqdm() as progress_bar:
        for i in range(steps):
            train(i)
            progress_bar.update()

    if save_z:
        if not os.path.exists(".\\latents"):
            Path.mkdir(".\\latents")

        torch.save(z, f"latents\{z_file}.pt")
except KeyboardInterrupt:
    pass
