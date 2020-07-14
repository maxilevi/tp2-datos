import re
import pandas as pd
import numpy as np
import collections
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
import math
import nltk

embedding_dim = 300 
embeddings_path = './data/embeddings/word2vec.bin'
embeddings = None

def process_dataset(df):
    df2 = df.copy()
    global feature_names
    
    df['keyword'] = df['keyword'].fillna('NAN_KEYWORD') #Reemplazo los Na con NAN_KEYWORD para poder agruparlos

    _add_text_embeddings(df2)
    _add_location_invalid_character_count_feature(df2)
    _add_length_features(df2)
    
    df2.drop(['text', 'location', 'keyword', 'target', 'id'], axis=1, inplace=True)
    return df2



def _add_text_embeddings(df):
    global embeddings_dim
    global embeddings
    global embeddings_path

    if embeddings is None:
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

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
        df[f'text_embedding_{i}'] = pd.Series(col)



def _add_length_features(df):
    def _length(x):
        return len(x) if type(x) is str else 0

    #stopwords = set(nltk.corpus.stopwords.words('english'))

    df['keyword_length'] = df['keyword'].map(_length)
    df['text_length'] = df['text'].map(_length)
    df['location_length'] = df['location'].map(_length)

    df['text_word_count'] = df['text'].map(lambda x: len(x.split(' ')))
#     df['stop_word_count'] = df_test['text'].map(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
#     df['punctuation_count'] = df['text'].map(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['hashtag_count'] = df['text'].map(lambda x: x.count('#'))
    df['mention_count'] = df['text'].map(lambda x: x.count('@'))
    df['char_count'] = df['text'].map(lambda x: len(x))
    df['exclamation_count'] = df['text'].map(lambda x: x.count('!'))
    df['interrogation_count'] = df['text'].map(lambda x: x.count('?'))
    df['url_count'] = df['text'].map(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df['word_count'] = df['text'].map(lambda x: len(str(x).split()))
    df['unique_word_count'] = df['text'].map(lambda x: len(set(str(x).split())))

#     andy
    df['word_density'] = df['text_word_count'] / (df['char_count'] + 1)
    df['capitals'] = df['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['num_unique_words'] = df['text'].apply(lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['text_word_count']
    
    df['keywords_mean_length_encoding'] = df.groupby('keyword')['text_length'].transform('mean')


def _add_location_invalid_character_count_feature(df):
    invalid_characters_regex = '#|\$|\|%|\?|!|/|;|@|\+|\*|\d'
    pattern = re.compile(invalid_characters_regex)
    
    def count_invalid_chars(x):
        if type(x) is float:
            return 0
        return len(re.findall(pattern, x))
    
    df['invalid_location_character_count'] = df['location'].map(count_invalid_chars)


def _clean_tweet(text):
    return text.lower() if type(text) is str else str()# temp
    text = re.sub('@.*?(\s*)', str(), text)
    text = re.sub('http(s?):\/\/.*\s*', str(), text)
    text = re.sub('\?*', str(), text)
    text = re.sub('OffensiveåÊContent', 'offensive content', text)
    text = re.sub('\n', str(), text)
    text = re.sub('#(.*?)(\s|$)', ' ', text)
    text = re.sub('\bdis\b', ' this ', text)
    text = re.sub('\bda\b', ' the ', text)
    return text.lower()
