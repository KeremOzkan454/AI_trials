import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

nltk.download("punkt")
nltk.download("stopwords")

# 1. Veri Seti Hazırlama
def get_data(data_file):
    with open(data_file, "r") as file:
        data = json.load(file)
    return data

data = get_data("intent_data.json")
df = pd.DataFrame(data, columns=["text", "intent"])

# 2. Tokenization ve Etiketleme
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

# Etiketleri sayısal değerlere dönüştür
df['label'] = df['intent'].astype('category').cat.codes

# Tokenize et
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Dataset'i oluştur
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# 3. Dataset Sınıfı
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels.tolist())
test_dataset = IntentDataset(test_encodings, test_labels.tolist())

# 4. Modeli Eğitme
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=len(df['label'].unique()))

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=test_dataset             
)

trainer.train()

# 5. Kullanıcı Mesajlarını Sınıflandırma
def predict_intent(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return df['intent'].astype('category').cat.categories[predictions.item()]

# Test: Kullanıcı Mesajı
while True:
    user_text = input(">>> ")
    intent = predict_intent(user_text)
    print(f"Kullanıcı niyeti: {intent}")