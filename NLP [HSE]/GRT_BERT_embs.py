from transformers import BertModel, BertTokenizer
import torch

# Загружаем BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Токенизируем текст
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Получаем эмбеддинги
with torch.no_grad():
    outputs = model(**inputs)

# Извлекаем скрытые состояния (эмбеддинги слов)
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # (1, N, 768)




from transformers import GPT2Model, GPT2Tokenizer

# Загружаем GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Токенизируем текст
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Получаем эмбеддинги
with torch.no_grad():
    outputs = model(**inputs)

# Эмбеддинги скрытых состояний
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # (1, N, 768)
