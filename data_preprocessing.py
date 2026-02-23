import pandas as pd
import json
from utils import get_tokenizer
import os
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import os
from model_args import Arguments

args = Arguments()


tokenizer = get_tokenizer()

def main():
    
    with open('archive/annotations_trainval2014/annotations/captions_train2014.json', 'r') as f:
        data = json.load(f)
    images_data = pd.DataFrame(data['images'])
    annotation_data = pd.DataFrame(data['annotations'])
    data = annotation_data.merge(images_data, how='left', left_on='image_id', right_on='id')
    data = data.drop(columns=['id_x', 'license', 'height', 'width', 'date_captured', 'flickr_url','coco_url', 'id_y', 'image_id'])
    
    data['file_name'] = data['file_name'].apply(lambda x: os.path.join('archive/train2014/train2014', x))
    
    # Save the processed data
    data.to_csv('processed_captions.csv', index=False)
    print("Saved to processed_captions.csv")
    
    data = load_dataset('csv', data_files='processed_caption.csv')
    data.map(tokenize, batch_size=5000, batched=True)
    
    data = data.map(tokenize, batched=5000, num_proc=3)
    


def transform(img_path):
    transformer = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = Image.open(img_path).convert('RGB')
    transformed_img = transformer(img)

    return transformed_img


def tokenize(sample):
    tokens = tokenizer(sample['caption'], padding='max_length', truncation=True, max_length=32, add_special_tokens=False)
    return {'input_ids': tokens['input_ids'], "attention_mask": tokens['attention_mask']}
    

    

if __name__ == "__main__":
    main()