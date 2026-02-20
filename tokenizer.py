from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "A dog is playing with a ball"
tokens = tokenizer(text)

print(tokens["input_ids"])
