import pandas as pd
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def load_data(path):
    df = pd.read_csv(path)
    df = df[['main_speaker', 'title', 'details']]
    df.dropna(inplace=True)
    df['details'] = df['title'] + ' ' + df['details']
    return df

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in text.split() if w.lower() not in stop_words]
    return " ".join(words)

def clean_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def preprocess_text(df):
    df['details'] = df['details'].apply(remove_stopwords)
    df['details'] = df['details'].apply(clean_punctuation)
    return df
