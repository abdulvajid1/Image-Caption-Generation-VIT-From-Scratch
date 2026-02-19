import torch
import torch.nn as nn
import torch.nn.functional as F
from model_args import Arguments
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_patches = (args.img_size//args.patch_size)**2
        self.positions = nn.Embedding(self.context_len, args.latent_dim)
    def forward(self, x):
        pos = self.positions(torch.arange(0, self.num_patches-1).to(torch.int))[None, :, :]
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
    
class LayerNorm(nn.Module): # Or RMS Norm
    def __init__(self, args):
        super().__init__()
        self.eps = args.eps
        self.scale = nn.Parameter(torch.ones(args.latent_dim))
        self.shift = nn.Parameter(torch.zeros(args.latent_dim))
    def forward(self, x: torch.Tensor):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        x_norm = (x - x_mean) / (x_std + self.eps)
        return self.scale * x_norm + self.shift
    
class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.latent_dim % args.num_heads == 0, "latent dim should be divisible with num heads"
        self.q_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=False)
        self.k_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=False)
        self.v_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=False)
        self.o_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=True)
    
    def forward(self, x: torch.Tensor):
        num_batch, seq_len, _ = x.size()
        x_q = self.q_proj(x) # (b, seq, latent_dim)
        x_k = self.k_proj(x)
        x_v = self.v_proj(x)
        
        head_dim = self.args.latent_dim // self.args.num_heads
        x_q = x_q.view(num_batch, seq_len, self.args.num_heads, head_dim).transpose(1, 2)
        x_k = x_k.view(num_batch, seq_len, self.args.num_heads, head_dim).transpose(1, 2)
        x_v = x_v.view(num_batch, seq_len, self.args.num_heads, head_dim).transpose(1, 2)
        
        attn_val = torch.matmul(x_q, x_k.transpose(-1, -2)) / (head_dim**0.5)
        attn_scores = F.softmax(attn_val, dim=-1)
        contexual_latent = torch.matmul(attn_scores, x_v)
        contexual_latent = contexual_latent.transpose(1, 2).contiguous().view(num_batch, seq_len, -1)
        
        return self.o_proj(contexual_latent)


class MLPLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        mid_out_features = args.latent_dim*args.latent_mull
        self.ff = nn.Sequential(
            nn.Linear(in_features=args.latent_dim, out_features=mid_out_features),
            nn.GELU(),
            nn.Linear(in_features=mid_out_features, out_features=args.latent_dim)
        )
    def forward(self, x):
        return self.ff(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.norm1 = LayerNorm(args)
        self.norm2 = LayerNorm(args)
        self.attention = MultiHeadAttention(args)
        self.mlp = MLPLayer(args)
    
    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn = self.attention(x_norm)
        x = x + x_attn
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        return x + x_mlp
        
        
    
        
        
        
    

    
    
    

if __name__ == "__main__":
    args = Arguments()
    pos = PositionalEmbedding(args)
    img_patch = ImagePatchEmbedding(args)
    x = torch.rand(5, args.img_channels, args.img_size, args.img_size)
    print(img_patch(x).shape)
        