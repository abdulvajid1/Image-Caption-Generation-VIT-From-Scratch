import torch
import torch.nn as nn
import torch.nn.functional as F
from model_args import Arguments
import math
from tokenizer import tokenizer


class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_patches = (args.img_size//args.patch_size)**2
        self.positions = nn.Embedding(self.num_patches, args.latent_dim)
    def forward(self, x):
        pos = self.positions(torch.arange(0, self.num_patches, device=x.device).to(torch.int))[None, :, :]
        return x + pos

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
    
class TextEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_tokens = args.context_len
        self.embed = nn.Embedding(num_embeddings=args.num_tokens, embedding_dim=args.latent_dim)
        self.pos_embed = nn.Embedding(num_embeddings=args.context_len, embedding_dim=args.latent_dim)
    
    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        seq_len = x.shape[1]
        return x + self.pos_embed(torch.arange(seq_len, device=x.device))[None, :seq_len, :]
        
        
    
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
        self.num_patches = (self.args.img_size//self.args.patch_size)**2
        mask = get_mask_val(self.num_patches, args.context_len)
        self.register_buffer('mask', mask)
        assert args.latent_dim % args.num_heads == 0, "latent dim should be divisible with num heads"
        self.q_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=False)
        self.k_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=False)
        self.v_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=False)
        self.o_proj = nn.Linear(in_features=args.latent_dim, out_features=args.latent_dim, bias=True)
    
    def forward(self, x: torch.Tensor):
        num_batch, seq_len, _ = x.size()
        text_token_len = seq_len - self.num_patches
        x_q = self.q_proj(x) # (b, seq, latent_dim)
        x_k = self.k_proj(x)
        x_v = self.v_proj(x)
        
        head_dim = self.args.latent_dim // self.args.num_heads
        x_q = x_q.view(num_batch, seq_len, self.args.num_heads, head_dim).transpose(1, 2)
        x_k = x_k.view(num_batch, seq_len, self.args.num_heads, head_dim).transpose(1, 2)
        x_v = x_v.view(num_batch, seq_len, self.args.num_heads, head_dim).transpose(1, 2)
        
        attn_val = torch.matmul(x_q, x_k.transpose(-1, -2)) / (head_dim**0.5)
        # TODO: Casual mask with image
        totel_mask_len = self.num_patches + text_token_len
        mask = self.mask[:, :, :totel_mask_len, :totel_mask_len] == 0
        attn_val.masked_fill_(mask, float('-inf'))
        
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
    
    
class VITBlock(nn.Module):
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

class VITHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.final_layer = nn.Linear(in_features=args.latent_dim, out_features=args.num_tokens)
        
    def forward(self, x):
        return self.final_layer(x)
        
class VIT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_input_layer = ImagePatchEmbedding(args)
        self.text_input_layer = TextEmbedding(args)
        self.vitblocks = nn.Sequential(*[VITBlock(args) for _ in range(args.num_layers)])
        self.output_layer = VITHead(args)
        self.patch_len = (args.img_size // args.patch_size)**2
    
    def forward(self, x_img, target_text_tokens:torch.Tensor=None, attn_mask=None):
        x = self.image_input_layer(x_img)
        loss = None
        if target_text_tokens != None:
            text_embeddings = self.text_input_layer(target_text_tokens)
            x = torch.concat((x, text_embeddings), dim=1)
        
        x = self.vitblocks(x)
        final_out = self.output_layer(x)
        if target_text_tokens != None:
            logits = final_out[:, self.patch_len-1 :-1, :].reshape(-1, self.args.num_tokens)
            # TODO: add end token, and -100 on padd token
            target_text_tokens = target_text_tokens.masked_fill(~attn_mask.to(bool), value=-100)
            
            targets = target_text_tokens.view(-1)
            loss = F.cross_entropy(logits, targets)
        return final_out, loss
    
    def vit_pass(self, x):
        x = self.vitblocks(x)  
        out = self.output_layer(x)
        return out
        
        
    @torch.no_grad()
    def generate(self, img, max_len=None):
        if not max_len:
            max_len = self.args.context_len
        if len(img.shape) < 4:
            img.unsqueeze_(0)
        batch_size = img.shape[0]
        img_embed = self.image_input_layer(img)
        curr_input = img_embed
        
        for i, _ in enumerate(range(max_len)):
            out = self.vit_pass(curr_input)
            last_layer_out = out[:, -1, :]
            next_tok_indices = torch.argmax(last_layer_out, dim=-1, keepdim=True)
            if i == 0:
                text_tokens = (torch.ones(size=(batch_size, 1), device=img.device) * next_tok_indices).to(torch.int)
            else: 
                text_tokens = torch.cat((text_tokens, next_tok_indices), dim=-1).to(img.device)
            text_embeds = self.text_input_layer(text_tokens)
            curr_input = torch.concat((img_embed, text_embeds), dim=1)
       
        return tokenizer.batch_decode(text_tokens.tolist())

# def get_mask_val(num_img_patches, text_token_len):
#     full_mask = torch.ones(num_img_patches+text_token_len, num_img_patches+text_token_len)
#     causal_mask = torch.tril(full_mask, diagonal=0)
#     causal_mask[:num_img_patches, :num_img_patches] = 1
#     return causal_mask.unsqueeze(0).unsqueeze(0)

def get_mask_val(num_img_patches, text_token_len):
    total = num_img_patches + text_token_len
    mask = torch.tril(torch.ones(total, total))
    mask[:num_img_patches, :num_img_patches] = 1  # image fully visible
    return mask.unsqueeze(0).unsqueeze(0)
    
    
        
# TODO : add attention correctly
# TODO : may need to add seperate token
    
    

if __name__ == "__main__":
    args = Arguments()
    Vit = VIT(args=args)
    x = torch.rand(5, args.img_channels, args.img_size, args.img_size)
    print(Vit(x))
        