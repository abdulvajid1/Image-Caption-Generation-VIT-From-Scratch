import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_patches = (args.img_size//args.patch_size)**2
        self.positions = nn.Parameter(torch.randn(1, self.num_patches, self.args.latent_dim))
    def forward(self, x):
        return x + self.positions

class ImagePatchEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pos_emb = PositionalEmbedding(args)
        self.net = nn.Conv2d(in_channels=args.img_channels, out_channels=args.latent_dim, kernel_size=args.patch_size, stride=args.patch_size, padding=0)
        
    def forward(self, img: torch.Tensor):
        x = self.net(img) # shape: (b, channels (latent), patch_size(2), patch_size(2))
        b, ch, _, _= x.size()
        x = x.view(b, ch, -1).permute(0, 2, 1) # (b, pathces, latent_dim)
        return self.pos_emb(x)
        