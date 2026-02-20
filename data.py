from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model_args import Arguments
from torchvision import transforms
from PIL import Image

args = Arguments()


class ImageCaptionData(Dataset):
    def __init__(self, path='processed_caption.csv'):
        super().__init__()
        self.path = path
        self.data = pd.read_csv(path)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return transform(self.data.iloc[index]['file_name']), self.data.iloc[index]['caption_tokens']
        

def get_dataloader():
    dataloader = DataLoader(dataset=ImageCaptionData(), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=args.pin_memory)
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