import pandas as pd
import json
from utils import get_tokenizer
import os

tokenizer = get_tokenizer()

def main():
    
    with open('archive/annotations_trainval2014/annotations/captions_train2014.json', 'r') as f:
        data = json.load(f)
    images_data = pd.DataFrame(data['images'])
    annotation_data = pd.DataFrame(data['annotations'])
    data = annotation_data.merge(images_data, how='left', left_on='image_id', right_on='id')
    data = data.drop(columns=['id_x', 'license', 'height', 'width', 'date_captured', 'flickr_url','coco_url', 'id_y', 'image_id'])
    
   
    # Apply tokenization to the caption column
    data['caption_tokens'] = data['caption'].apply(lambda x: tokenize({'caption': x}))
    print("Tokenization complete!")
    
    data['file_name'] = data['file_name'].apply(lambda x: os.path.join('archive/train2014/train2014', x))
    
    # Save the processed data
    data.to_csv('processed_captions.csv', index=False)
    print("Saved to processed_captions.csv")
    

def tokenize(sample):
        # Tokenize the caption column
        tokens = tokenizer(sample['caption'], padding='max_length', truncation=True, max_length=128)
        return tokens['input_ids']
    

if __name__ == "__main__":
    main()