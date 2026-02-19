from dataclasses import dataclass

@dataclass
class Arguments:
    latent_dim: int = 256
    patch_size: int = 16
    img_size: int = 64
    img_channels: int = 3
    eps: float = 1e-6
    num_heads: int = 8
    