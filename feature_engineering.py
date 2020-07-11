import pandas as pd
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('stopwords')

tweets = pd.read_csv('train.csv')

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    ''''''
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

text_cleaner = np.vectorize(normalize_document)

# Funcion extraída de https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41#:~:text=The%20importance%20of%20feature%20engineering,understood%20by%20machine%20learning%20algorithms.

def feature_engineering(original_df):
    '''Devuelve una copia del df con nuevos features [location_length, keyword_count, len], además de columnas resultado de tf-idf'''
    df = original_df.copy()
    
    clean_text = text_cleaner(df['text'])
    df['text'] = clean_text
    
    df = keyword_counter(df)
    location_length_counter(df)
    text_lenght_agg(df)
    convert_NaN_to_0(df)


    
    return df[['len', 'keyword_count', 'location_length']]
    
def keyword_counter(df):
    keywords = df['keyword']
    keyword_count = keywords.value_counts()
    keyword_count = keyword_count.to_frame().reset_index()
    keyword_count.columns = ['keyword', 'keyword_count']
    df2 = df.merge(keyword_count, how='left', on='keyword')
    
    return df2

def location_length_counter(df):
    df['location_length'] = [(0 if isinstance(t, float) else len(t)) for t in df['location']]
    

def text_lenght_agg(df):
    df['len'] = df['text'].transform(lambda x : len(x))
    
def convert_NaN_to_0(df):
    for col_name in df.columns:
        df[col_name] = df[col_name].fillna(0)