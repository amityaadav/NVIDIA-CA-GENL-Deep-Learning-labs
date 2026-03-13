import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
print(indexed_tokens)