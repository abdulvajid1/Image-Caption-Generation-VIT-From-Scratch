# Image Caption Generation with Vision Transformer (ViT) from Scratch

This project is a hands-on reimplementation of image captioning using a pure Vision Transformer (ViT) pipeline instead of the classic CNN encoder + LSTM decoder setup. The goal is to explore how a transformer-based encoder and decoder can be built from scratch for image-to-text generation.

The original idea was to recreate an earlier architecture based on:

- CNN encoder
- LSTM decoder

but this version replaces both with transformer blocks and patch-based visual tokens.

## Why this project exists

This repository is a learning project focused on understanding the full pipeline of image captioning with transformers:

- image patch embedding
- positional encoding
- token embedding for captions
- causal masking for text generation
- autoregressive decoding
- transformer-based feature learning

It is intentionally implemented from scratch so the architecture is easy to inspect and understand.

## Project overview

Image captioning is the task of generating a natural language description for an image. In this project:

- the image is split into fixed-size patches
- each patch is projected into a latent embedding space
- the image tokens are passed through a stack of transformer blocks
- text tokens are embedded and processed with positional information
- the model predicts the next caption token autoregressively

The model is trained on COCO-style image-caption data and uses a tokenizer to convert text into token IDs.

## Architecture summary

The implemented architecture follows a ViT-style encoder and a decoder-style transformer head for caption generation.

### Main components

- ImagePatchEmbedding
  - takes image tensor
  - divides it into patches using a convolutional patch projector
  - adds positional embeddings

- TextEmbedding
  - embeds caption tokens
  - adds learned positional embeddings

- MultiHeadAttention
  - standard multi-head self-attention
  - uses an attention mask to preserve causal behavior

- MLPLayer
  - feed-forward block after attention
  - applies GELU activation

- VITBlock
  - layer norm + attention + residual connection
  - layer norm + MLP + residual connection

- VIT
  - combines image and text embeddings
  - applies transformer blocks
  - outputs logits over token vocabulary

### Transformer configuration used in this repo

From `model_args.py`:

- latent_dim: 1024
- patch_size: 16
- img_size: 224
- img_channels: 3
- num_heads: 8
- context_len: 32
- num_layers: 14
- num_tokens: 30522
- latent_mull: 4
- learning_rate: 2e-5
- batch_size: 32

This means that for a 224x224 RGB image:

- image is split into patches of size 16x16
- total patches = (224 / 16) ^ 2 = 196
- each patch becomes a latent vector of size 1024

That gives a sequence length of 196 image tokens plus caption tokens, which is then passed through transformer layers.

## Data pipeline

The repository uses the COCO dataset and prepares caption CSV files before training.

The file `data_preprocessing.py`:

- loads the COCO annotations JSON
- merges image metadata with caption annotations
- cleans the dataset fields
- modifies the image path to point to train/val dataset directories
- tokenizes captions using the project tokenizer
- saves processed datasets to disk

The key preprocessing steps are:

- caption tokenization with `max_length = 32`
- labels are generated with padding masks
- image preprocessing includes resize and normalization

## Training pipeline

The main training loop is in `train.py`.

It does the following:

- instantiates the model (`VIT(args)`) on CUDA
- loads AdamW optimizer
- loads training and validation dataloaders
- logs to TensorBoard
- evaluates every `eval_step`
- saves checkpoints every `save_step`
- uses mixed precision (`bfloat16`) during training
- clips gradients with `clip_grad_norm_`

The training logic follows a standard supervised autoregressive captioning setup:

- input: image tensor + previous text tokens
- target: next caption token prediction
- loss: cross entropy over vocabulary

## Generation / inference

The model includes a `generate()` function in `vit.py`.

This function:

- encodes the image into patch embeddings
- starts with a generated token sequence
- iteratively predicts the next token
- appends the token to the sequence
- continues until the context length is reached

This produces a caption in an autoregressive fashion, similar to decoder-only text generation.

## Repository files

- `vit.py` – model definition and generation logic
- `model_args.py` – hyperparameter settings
- `train.py` – training loop and evaluation
- `data_preprocessing.py` – COCO preprocessing and tokenization
- `data.py` – dataloader and dataset utilities
- `utils.py` – model saving/loading helpers
- `tokenizer.py` – tokenizer setup
- `test.ipynb` – exploratory testing / generation notebook
- `pyproject.toml` – project config and dependency list

## Dependencies

The project relies on:

- PyTorch
- TorchVision
- Hugging Face Transformers
- Datasets
- Pandas
- Pillow
- Loguru

These dependencies are listed in `pyproject.toml`.

## How to run

### 1. Install dependencies

With `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install datasets pandas pillow torch torchvision transformers loguru
```

### 2. Prepare the dataset

```bash
python data_preprocessing.py
```

### 3. Train the model

```bash
python train.py
```

### 4. Run notebook experiments

Open `test.ipynb` for quick experimentation and result checking.

## Notes on the architecture

This project is intentionally experimental and educational. There are a few engineering details worth knowing:

- the model uses a custom ViT-like architecture rather than a standard pretrained backbone
- the attention mask is used to control which tokens are visible to which tokens
- the implementation is built for research and experimentation rather than production deployment
- the decoder is autoregressive and learns to predict the next caption token

## Strengths of this implementation

- end-to-end transformer-based image caption model
- custom implementation easy to modify and inspect
- patch-based vision representation instead of CNN feature maps
- direct adaptation of a classic image captioning setup to a transformer paradigm

## Limitations / TODOs

This repo is a learning project and still has room for improvement:

- better caption generation decoding methods such as beam search
- more robust evaluation metrics like BLEU, METEOR, CIDEr
- cleaner separation between encoder and decoder responsibilities
- stronger causal attention logic for decoder-side generation
- more careful handling of training stability and validation metrics

## Future directions

Possible next steps for this project include:

- switching from greedy decoding to beam search
- adding a pretrained ViT backbone for stronger feature extraction
- evaluating the model with standard caption metrics
- integrating a more explicit encoder-decoder design
- building a cleaner training/evaluation pipeline for reproducibility

## Summary

This project represents a meaningful step away from the classic CNN + LSTM captioning design and toward a full transformer-based image captioning model built from scratch. It is a strong example of rethinking multimodal generation through patch embeddings, self-attention, and autoregressive caption generation.

The result is a compact but expressive model that is easy to study, modify, and extend.

---

If you'd like, I can also help with any of the following:

- improve the model architecture further
- add a better README with diagrams and training instructions
- create a cleaner `requirements.txt`
- add a demonstration notebook for generating captions from example images
- help debug or optimize the training loop
