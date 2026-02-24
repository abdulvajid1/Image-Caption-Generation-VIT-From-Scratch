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
from utils import load_model

from loguru import logger

def generate_random(model, eval_path="archive/val2017/val2017"):
    val_imgs = list(Path(eval_path).glob("*.jpg"))
    rand_int = random.randint(0, 1000)
    rand_img_path = val_imgs[rand_int]
    img = transform(rand_img_path)
    model.eval()
    print(model.generate(img))
    
    pass
def train(model: torch.nn.Module, optimizer: torch.optim.AdamW, dataloader: torch.utils.data.DataLoader, epoch_num: int, global_step):
    progress_bar = tqdm.tqdm(dataloader)
    
    for step, (img, text_tokens) in enumerate(dataloader):
        _, loss = model(img, text_tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch_num+1 % 15 == 0:
            logger.info("Generating Random Sentence")
            generate_random(model=model)
        
        progress_bar.set_postfix({
            "loss" : loss.item()
            })
        
        
        
    

def main():
    args = Arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VIT(args).to(device)
    # Then in your code:
    logger.info("model initialized")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataloader = get_dataloader(device)
    global_step = 0
    if args.load:
        global_step = load_model(model, optimizer)
     
      
    for epoch in range(args.num_epochs):
        logger.info(f"Starting training epoch {epoch}") 
        train(model, optimizer, dataloader, epoch, global_step)
        
        
if __name__ == '__main__':
    main()