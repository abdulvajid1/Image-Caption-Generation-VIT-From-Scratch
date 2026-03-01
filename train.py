import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import VIT
from model_args import Arguments
from data import get_dataloader, transform
import tqdm
from pathlib import Path
import random
from PIL import Image
from utils import load_model, save_model

from loguru import logger

torch.set_float32_matmul_precision('high')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

def generate_random(model, train=True):

    if train:
        path = "archive/train2014/train2014"
        logger.info('Eval from Train image')
    else:
        path="archive/val2017/val2017"
        logger.info('Eval from Train image')
    
    val_imgs = list(Path(path).glob('*.jpg'))
    rand_int = random.randint(0, 1000)
    rand_img_path = val_imgs[rand_int]
    logger.info(f"Testing with random image {rand_img_path}")
    img = transform(rand_img_path).to(device)
    model.eval()
    print(model.generate(img))
    return 0

def train(model: torch.nn.Module, optimizer: torch.optim.AdamW, dataloader: torch.utils.data.DataLoader, epoch_num: int, global_step, eval_step, save_step):
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch: {epoch_num}")
    model.train()
    for step, (img, text_tokens, attn_mask) in enumerate(progress_bar):
        img = img.to(device)
        text_tokens = text_tokens.to(device)
        attn_mask = attn_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(img, text_tokens, attn_mask)
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (global_step+1) % eval_step == 0:
            logger.info("Generating Random Sentence")
            model.eval()
            generate_random(model=model)
            generate_random(model=model, train=False)
            model.train()
        
        progress_bar.set_postfix({
            "loss" : loss.item()
            })
        

        if (global_step+1) % save_step == 0:
            save_model(model, optimizer, global_step)
            logger.info(f"Model Saved at {global_step}")
        global_step+=1

    return global_step

def main():
    args = Arguments()
    model = VIT(args).to(device)
    # model.compile()
    # Then in your code:
    logger.info("model initialized")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    dataloader = get_dataloader(device)
    global_step = 0
    if args.load:
        global_step = load_model(model, optimizer)
     
      
    for epoch in range(args.num_epochs):
        logger.info(f"Starting training epoch {epoch}") 
        global_step = train(model, optimizer, dataloader, epoch, global_step, args.eval_step, args.save_step)
        
        
if __name__ == '__main__':
    main()
    # Todo: mixed precision
    # attention
    # more batch size