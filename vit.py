import torch
import torch.nn as nn
import torch.nn.functional as F
from model_args import Arguments

class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_patches = (args.img_size//args.patch_size)**2
        self.positions = nn.Embedding(self.num_patches, args.latent_dim)
    def forward(self, x):
        pos = self.positions(torch.range(0, self.num_patches-1).to(torch.int))[None, :, :]
        print("pos embedding shape", pos.shape, "and shape of patches embedding", x.shape)
        return x + pos

class ImagePatchEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pos_emb = PositionalEmbedding(args)
        self.net = nn.Conv2d(in_channels=args.img_channels, out_channels=args.latent_dim, kernel_size=args.patch_size, stride=args.patch_size, padding=0)
        
    def forward(self, img: torch.Tensor):
        print()
        x = self.net(img) # shape: (b, channels (latent), patch_size(2), patch_size(2))
        b, ch, _, _= x.size()
        x = x.view(b, ch, -1).permute(0, 2, 1) # (b, pathces, latent_dim)
        return self.pos_emb(x)
    
    

if __name__ == "__main__":
    args = Arguments()
    pos = PositionalEmbedding(args)
    img_patch = ImagePatchEmbedding(args)
    x = torch.rand(5, args.img_channels, args.img_size, args.img_size)
    print(img_patch(x).shape)
        