import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import json

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("punkt")

# 1. Veri Seti Hazırlama
def get_data(data_file):
    with open(data_file,"r") as file:
        data = json.load(file)
    return data

data = get_data("intent_data.json")

df = pd.DataFrame(data, columns=["text", "intent"])


# 2. Stopword'leri Temizleme ve Tokenization
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("turkish"))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Küçük harfe çevir ve tokenleştir
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

# 3. Özellik Çıkarımı (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["intent"]

# 4. Modeli Eğitme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Doğruluğu
accuracy = model.score(X_test, y_test)
print(f"Model Doğruluğu: {accuracy:.2f}")


# 5. Kullanıcı Mesajlarını Sınıflandırma
def predict_intent(user_input):
    processed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    prediction = model.predict(vectorized_input)
    return prediction[0]

"""
# Test: Kullanıcı Mesajı
while True:
    user_text = input(">>> ")
    intent = predict_intent(user_text)
    print(f"Kullanıcı niyeti {intent}")
