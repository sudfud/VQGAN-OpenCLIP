import generate

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_from = torch.load("latents\\mushroom.pt")
img_to = torch.load("latents\\planets.pt")

