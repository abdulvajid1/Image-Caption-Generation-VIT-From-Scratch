from dataclasses import dataclass

@dataclass
class Arguments:
    latent_dim: int = 1024
    patch_size: int = 16
    img_size: int = 224
    img_channels: int = 3
    eps: float = 1e-8
    num_heads: int = 8
    context_len: int = 32
    batch_size: int = 32
    num_workers: int = 6
    pin_memory: bool = True
    load: bool = False
    # num_patches: int = 16
    num_tokens: int = 30522
    num_layers: int = 12
    latent_mull: int = 4
    learning_rate: int = 5e-5
    num_epochs: int = 20
    eval_step: int = 1000
    save_step: int = 1000
    save_path: str = "checkpoints"