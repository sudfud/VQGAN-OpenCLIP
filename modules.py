import sys
sys.path.append("taming-transformers")

import kornia.augmentation as K

from omegaconf import OmegaConf

from PIL import Image

from taming.models import cond_transformer, vqgan

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as pl

import math

# Normalized sinc function: sin(pi * n) / pi * n if n != 0, otherwise 1
def sinc(x: Tensor) -> Tensor:
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x: Tensor, a: int) -> Tensor:
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio: float, width: float) -> Tensor:
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0

    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio

    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input: Tensor, size: tuple[int, int], align_corners: bool = True) -> Tensor:
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    
    return F.interpolate(input, size, mode="bicubic", align_corners = align_corners)

class ReplaceGrad(autograd.Function):
    @staticmethod
    def forward(ctx: autograd.function.FunctionCtx, x_forward: Tensor, x_backward: Tensor) -> Tensor:
        ctx.shape = x_backward.shape
        return x_forward
    
    @staticmethod
    def backward(ctx, grad_in: Tensor) -> Tensor:
        return None, grad_in.sum_to_size(ctx.shape)

class ClampWithGrad(autograd.Function):
    @staticmethod
    def forward(ctx: autograd.function.FunctionCtx, input: Tensor, min: int, max: int):
        ctx.min = min
        ctx.max = max

        ctx.save_for_backward(input)

        return input.clamp(min, max)
    
    @staticmethod
    def backward(ctx: autograd.function.FunctionCtx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
    
def vector_quantize(x: Tensor, codebook: Tensor) -> Tensor:
    d = x.pow(2).sum(dim = -1, keepdim = True) + codebook.pow(2).sum(dim = 1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook

    return ReplaceGrad.apply(x_q, x)

class Prompt(nn.Module):
    def __init__(self, embed: Tensor, weight: float = 1.0, stop: float = float("-inf")):
        super().__init__()

        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input: Tensor) -> Tensor:
        input_normal = F.normalize(input.unsqueeze(1), dim = 2)
        embed_normal = F.normalize(self.embed.unsqueeze(0), dim = 2)

        dists = torch.linalg.norm(input_normal.sub(embed_normal), dim = 2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()

        return self.weight.abs() * ReplaceGrad.apply(dists, torch.maximum(dists, self.stop)).mean()
    
def parse_prompt(prompt: str) -> tuple[str, float, float]:
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', "-inf"][len(vals):]
    
    return vals[0], float(vals[1]), float(vals[2])
    
class MakeCutouts(nn.Module):
    def __init__(self, cut_size: float, cutn: int, cut_pow: float = 1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p = 0.5),
            K.RandomSharpness(0.3, p = 0.4),
            K.RandomAffine(degrees = 30, translate = 0.1, p = 0.8, padding_mode = "border"),
            K.RandomPerspective(0.2, p = 0.4),
            K.ColorJitter(hue = 0.01, saturation = 0.01, p = 0.7)
        )

        self.noise_fac = 0.1

    def forward(self, input: Tensor) -> Tensor:
        sideX, sideY = input.shape[2:4]

        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)

        cutouts = []

        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)

            offsetX = torch.randint(0, sideX - size + 1, ())
            offsetY = torch.randint(0, sideY - size + 1, ())

            cutout = input[:, :, offsetY:offsetY + size, offsetX:offsetX + size]

            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        batch = self.augs(torch.cat(cutouts, dim = 0))

        if self.noise_fac:
            factors = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + factors * torch.randn_like(batch)

        return batch
        
def load_vqgan_model(config_path: str, checkpoint_path: str) -> pl.LightningModule:
    config = OmegaConf.load(config_path)

    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)

    else:
        raise ValueError(f"Unknown model type: {config.model.target}")
    
    del model.loss

    return model

def resize_image(image: Image, out_size: tuple[int, int]) -> Image:
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)

    return image.resize(size, Image.LANCZOS)