import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# NLTK veri setlerini indir
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_text(text):
    # Küçük harflere çevir
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize et
    tokens = word_tokenize(text)
    # Stop words (anlamsız kelimeleri) kaldır
    stop_words = set(stopwords.words('turkish'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Örnek
while True:
    sample_text = input(">>> ")
    print(preprocess_text(sample_text))
