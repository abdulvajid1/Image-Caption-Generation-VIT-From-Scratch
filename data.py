from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model_args import Arguments
from torchvision import transforms
from PIL import Image
import torch
from datasets import load_from_disk

args = Arguments()


class ImageCaptionData(Dataset):
    def __init__(self, path='caption_data', device='cpu'):
        super().__init__()
        self.path = path
        self.data = load_from_disk(path)['train']
        self.data.set_format('torch')
        self.device = device
        
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return transform(self.data[index]['file_name']).to(self.device), (self.data[index]['input_ids']).to(self.device)
    
    @staticmethod
    def transform(img_path):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        img = Image.open(img_path)
        transformed_img = transformer(img)
        return transformed_img
        

def get_dataloader(device):
    dataloader = DataLoader(dataset=ImageCaptionData(device=device), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=args.pin_memory)
    return dataloader


def transform(img_path):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    
    img = Image.open(img_path)
    transformed_img = transformer(img)
    
    return transformed_img