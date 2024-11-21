import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image

def gaussian_noise(x, severity=1, device='cuda'):
    c = [.03, .05, .1, .2, .3][severity - 1]
    noise = torch.randn_like(x).to(device) * c
    return torch.clamp(x + noise, 0, 1)

def shot_noise(x, severity=1, device='cuda'):
    c = [600, 100, 50, 12, 5][severity - 1]
    noise = torch.poisson(x * c) / float(c)
    return torch.clamp(noise, 0, 1)

def impulse_noise(x, severity=1, device='cuda'):
    c = [.01, .02, .03, .09, .17][severity - 1]
    noise = torch.rand_like(x).to(device)
    mask = noise < c
    x = x * (~mask) + torch.rand_like(x).to(device) * mask
    return torch.clamp(x, 0, 1)

def speckle_noise(x, severity=1, device='cuda'):
    c = [.03, .04, .05, .1, .2][severity - 1]
    noise = torch.randn_like(x).to(device) * c
    return torch.clamp(x + x * noise, 0, 1)

def gaussian_blur(x, severity=1, device='cuda'):
    c = [0.3, 0.5, 0.6, 1, 2][severity - 1]
    gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(c, c))
    return gaussian_blur(x)

def zoom_blur(x, severity=1, device='cuda'):
    c = [torch.arange(1, 1.05, 0.01),
         torch.arange(1, 1.11, 0.01),
         torch.arange(1, 1.16, 0.02),
         torch.arange(1, 1.21, 0.02),
         torch.arange(1, 1.25, 0.03)][severity - 1]

    out = torch.zeros_like(x)
    for zoom_factor in c:
        zoomed = F.interpolate(x, scale_factor=zoom_factor.item(), mode='bilinear', align_corners=False)
        out += zoomed[:, :, :x.size(2), :x.size(3)]
    return torch.clamp((x + out) / (len(c) + 1), 0, 1)

def snow(x, severity=1, device='cuda'):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
    
    snow_layer = torch.normal(mean=c[0], std=c[1], size=(x.size(0), 1, x.size(2), x.size(3))).to(device)
    snow_layer = F.interpolate(snow_layer, scale_factor=c[2], mode='bilinear', align_corners=False)
    snow_layer = snow_layer * (snow_layer > c[3]).float()
    
    return torch.clamp(x + snow_layer * c[6], 0, 1)

def contrast(x, severity=1, device='cuda'):
    c = [.7, .3, .2, .1, .05][severity - 1]
    means = torch.mean(x, dim=(2, 3), keepdim=True)
    return torch.clamp((x - means) * c + means, 0, 1)

def brightness(x, severity=1, device='cuda'):
    c = [.1, .2, .3, .4, .5][severity - 1]
    transform = transforms.ColorJitter(brightness=c)
    return transform(x)

def saturate(x, severity=1, device='cuda'):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    transform = transforms.ColorJitter(saturation=c[0])
    return transform(x)

def pixelate(x, severity=1, device='cuda'):
    c = [0.3, 0.25, 0.15, 0.1, 0.05][severity - 1]
    x = F.interpolate(x, scale_factor=c, mode='bilinear', align_corners=False)
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)