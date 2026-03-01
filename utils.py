
from pathlib import Path
import torch
from loguru import logger
from model_args import Arguments
import os

args = Arguments()

def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def load_model(model, optimizer, global_step=None):
    if not global_step:
        ckpt_paths = list(Path(args.save_path).glob('*.pt'))
        ckpt = max(ckpt_paths, key=os.path.getatime)

    logger.info(f"Loading model from {ckpt}")
    ckpt_dict = torch.load(ckpt)
    model.load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    return int(ckpt.as_posix().split("_")[-1][:-3])


def save_model(model,  optimizer, global_step):
    checkpoint_path = Path(args.save_path)
    checkpoint_path.mkdir(exist_ok=True)
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    path_to_save = checkpoint_path.joinpath(f'ckpt_{global_step}.pt')
    torch.save(save_dict, path_to_save.as_posix())
    