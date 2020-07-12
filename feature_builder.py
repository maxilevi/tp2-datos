import re
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

embedding_dim = 300 
embeddings_path = './data/embeddings/word2vec.bin'
embeddings = None
feature_names = ['invalid_location_character_count', 'keyword_length', 'text_length', 'location_length'] + [f'embedding_{i}' for i in range(embedding_dim)]

def process_dataset(df):
    df2 = df.copy()
    global embeddings
    global embeddings_path
    global feature_names

    if embeddings is None:
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

    _add_text_embeddings(df2)
    _add_location_invalid_character_count_feature(df2)
    _add_length_features(df2)
    
    return df2[feature_names]
    

def _add_text_embeddings(df):
    global embeddings_dim
    global embeddings
    text_values = [ _clean_tweet(x) for x in df['text'].values]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_values)
    inv_word_index = {v: k for k, v in tokenizer.word_index.items()}
    as_sequences = tokenizer.texts_to_sequences(text_values)

    embeddings_rows = []
    for j in range(len(as_sequences)):
        avg_embedding = np.zeros(embedding_dim)
        count = 0
        for k in range(len(as_sequences[j])):
            index = as_sequences[j][k]
            word = inv_word_index[index]
            if word in embeddings:
                avg_embedding += embeddings[word]
                count += 1

        avg_embedding = avg_embedding / count if count > 0 else avg_embedding
        embeddings_rows.append(avg_embedding)
    

    for i in range(embedding_dim):
        col = []
        for j in range(len(embeddings_rows)):
            col.append(embeddings_rows[j][i])
        df[f'embedding_{i}'] = pd.Series(col)


def _add_length_features(df):
    def _length(x):
        return len(x) if type(x) is str else 0

    df['keyword_length'] = df['keyword'].map(_length)
    df['text_length'] = df['text'].map(_length)
    df['location_length'] = df['location'].map(_length)
    df['keywords_mean_length_encoding']=df.groupby('keyword')['text_length'].transform('mean')


def _add_location_invalid_character_count_feature(df):
    invalid_characters_regex = '#|\$|\|%|\?|!|/|;|@|\+|\*|\d'
    pattern = re.compile(invalid_characters_regex)
    
    def count_invalid_chars(x):
        if type(x) is float:
            return 0
        return len(re.findall(pattern, x))
    
    df['invalid_location_character_count'] = df['location'].map(count_invalid_chars)


def _clean_tweet(text):
    return text.lower()# temp
    text = re.sub('@.*?(\s*)', str(), text)
    text = re.sub('http(s?):\/\/.*\s*', str(), text)
    text = re.sub('\?*', str(), text)
    text = re.sub('OffensiveåÊContent', 'offensive content', text)
    text = re.sub('\n', str(), text)
    text = re.sub('#(.*?)(\s|$)', ' ', text)
    text = re.sub('\bdis\b', ' this ', text)
    text = re.sub('\bda\b', ' the ', text)
    return text.lower()
