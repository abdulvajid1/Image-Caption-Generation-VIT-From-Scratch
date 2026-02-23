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



def generate_random(model, eval_path="archive/val2017/val2017"):
    val_imgs = list(Path(eval_path).glob("*.jpg"))
    rand_int = random.randint(0, 1000)
    rand_img_path = val_imgs[rand_int]
    img = transform(rand_img_path)
    model.eval()
    print(model.generate(img))
    
    pass
def train(model: torch.nn.Module, optimizer: torch.optim.AdamW, dataloader: torch.utils.data.DataLoader, epoch_num):
    progress_bar = tqdm.tqdm(dataloader)
    
    for step, (img, text_tokens) in enumerate(dataloader):
        _, loss = model(img, text_tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
    

def main():
    args = Arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VIT(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(x), lr=args.learning_rate)
    dataloader = get_dataloader(device)
    global_step = 0
    if args.load:
        global_step = load_model(model, optimizer)
        
    for epoch in args.num_epochs:
        train(model, optimizer, dataloader, epoch, global_step)