from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5-small") # tokeneizer is defined for the function as t5-small is used it is loaded here

def tokenize_sample(sample):
    tokenized_sample = tokenizer(sample['input'], text_target=sample['reference'], padding=True)
    tokenized_sample['labels'] = tokenized_sample['labels']
    return tokenized_sample
