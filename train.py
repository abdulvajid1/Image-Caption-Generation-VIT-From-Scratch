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
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


torch.set_float32_matmul_precision('high')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

@torch.no_grad()
def evaluate(model, dataloader, writer: SummaryWriter, global_step, num_step=20):
    model.eval()
    progress_bar = tqdm.tqdm(dataloader)
    total_loss = []
    for step, (img, text_tokens, attn_mask, filename) in enumerate(progress_bar):
        img = img.to(device)
        text_tokens = text_tokens.to(device)
        attn_mask = attn_mask.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(img, text_tokens, attn_mask)
        
        total_loss.append(loss.item())

        if (step+1) % num_step == 0:
            break
    
    writer.add_scalar(tag='eval loss', scalar_value=torch.tensor(total_loss).mean().item(), global_step=global_step)
        

def generate_random(model, train=True):

    if train:
        path = "archive/train2014/train2014"
        logger.info('Eval from Train image')
    else:
        path="archive/val2017/val2017"
        logger.info('Eval from Eval image')
    
    val_imgs = list(Path(path).glob('*.jpg'))
    rand_int = random.randint(0, 1000)
    rand_img_path = val_imgs[rand_int]
    logger.info(f"Testing with random image {rand_img_path}")
    img = transform(rand_img_path).to(device)
    model.eval()
    return  model.generate(img)

def train(model: torch.nn.Module, optimizer: torch.optim.AdamW, dataloader: torch.utils.data.DataLoader, eval_dataloader: torch.utils.data.DataLoader, epoch_num: int, global_step, eval_step, save_step, writer: SummaryWriter):
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch: {epoch_num}")
    model.train()
    total_loss = []
    for step, (img, text_tokens, attn_mask, filename) in enumerate(progress_bar):
        img = img.to(device)
        text_tokens = text_tokens.to(device)
        attn_mask = attn_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(img, text_tokens, attn_mask)

        # if loss.item() > 4:
        #     continue
        # _, loss = model(img, text_tokens, attn_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        writer.add_scalar('loss_per_step', loss.item(), global_step=global_step)
        total_loss.append(loss.item())

        # Evaluate the model in tensorboard
        if (global_step+1) % eval_step == 0:
            logger.info("Evaluating...")
            model.eval()
            evaluate(model, eval_dataloader, writer, global_step=global_step)
            writer.add_scalar(tag='Train loss', scalar_value=torch.tensor(total_loss).mean().item(), global_step=global_step)
            print(generate_random(model=model, train=True)[0])
            print(generate_random(model=model, train=False)[0])
            model.train()
    

        if (global_step+1) % save_step == 0:
            save_model(model, optimizer, global_step)
            logger.info(f"Model Saved at {global_step}")
        global_step+=1

        progress_bar.set_postfix({
            'loss': loss.item()
        })

    return global_step

def main():
    args = Arguments()
    model = VIT(args).to(device)
    model.compile()
    # Then in your code:
    logger.info("model initialized")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataloader = get_dataloader(device, train=True)
    eval_dataloader = get_dataloader(device, train=False)
    writer = SummaryWriter(log_dir=f'runs/ImageCaptioninig_lr_{args.learning_rate}_bz_{args.batch_size}_img_{args.img_channels}')
    global_step = 0
    if args.load:
        global_step = load_model(model, optimizer)
     
      
    for epoch in range(args.num_epochs):
        logger.info(f"Starting training epoch {epoch}") 
        global_step = train(model, optimizer, dataloader,eval_dataloader, epoch, global_step, args.eval_step, args.save_step, writer)
        
        
if __name__ == '__main__':
    main()
    # Todo: mixed precision
    # attention
    # more batch size