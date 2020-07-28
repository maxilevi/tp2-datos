import re
import pandas as pd
import numpy as np
import collections
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
import math
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import spacy
import os

embedding_dim = 300 
uncompressed_embeddings_path = '../data/embeddings/word2vec-uncompressed'
embeddings_path = '../data/embeddings/word2vec.bin'
embeddings = None
mean_encodings = None
spacy_nlp = None

def process_dataset(df, encoding_type='binary', text_type='embeddings', target_dimensions=None, clean_text=False, use_spacy=True, use_manual_features=True, remove_target=True):
    df2 = df.copy()
    global feature_names

    if use_manual_features:
        add_location_features(df2)
    calculate_keyword_encoding(df2, encoding_type=encoding_type)
    if use_manual_features:
        add_manual_text_features(df2)

    text_values = df2['text'].values
    if clean_text:
        text_values = [_clean_tweet(x) for x in text_values]
    
    if text_type == 'embeddings':
        df2 = add_text_embeddings(df2, text_values=text_values)
        if use_spacy:
            df2 = add_text_embeddings(df2, text_values=_generate_spacy_text_values(df2), prefix='spacy_')

    elif text_type == 'tfidf':
        add_text_tfidf(df2, text_values)

    elif text_type == 'bow':
        add_text_bow(df2, text_values)

    elif text_type == 'none':
        pass
    
    df2.drop(['text', 'location', 'keyword', 'id'], axis=1, inplace=True)
    if 'target' in df2.columns and remove_target:
        df2.drop(['target'], axis=1, inplace=True)

    if target_dimensions:
        df2 = reduce_dimensions(df2, dims=target_dimensions)

    return df2


def _generate_spacy_text_values(df):
    global spacy_nlp

    if not spacy_nlp:
        spacy_nlp = spacy.load("en_core_web_sm")

    def _process_tweet_spacy(x):
        tokens = [t.text for t in spacy_nlp(x) if t.pos_ in ['VERB', 'NOUN', 'ADJ', 'PROPN']]
        return ' '.join(tokens)
    return [_process_tweet_spacy(x) for x in df['text'].values]


def _add_text_using_vectorizer(df, vectorizer, text_values):    
    matrix = vectorizer.fit_transform(text_values)
    feature_names = vectorizer.get_feature_names()

    results_df = pd.DataFrame(matrix.T.todense(), index=feature_names, columns=[n for n in range(len(text_values))])
    for i in range(len(feature_names)):
        feature = feature_names[i]
        feature_column = results_df.loc[feature, :]
        if feature_column.sum() >= 0.5:
            df[feature] = results_df.loc[feature, :]



def add_text_bow(df, text_values):
    _add_text_using_vectorizer(df, CountVectorizer(ngram_range=(1, 3)), text_values)
    

def add_text_tfidf(df, text_values):
    _add_text_using_vectorizer(df, TfidfVectorizer(ngram_range=(1, 3)), text_values)


def reduce_dimensions(df, dims):
    pca = PCA(n_components=dims)
    matrix = pca.fit_transform(df)
    new_df = pd.DataFrame(data=matrix, columns=[f'dim_{i}' for i in range(dims)])
    return new_df


def add_text_embeddings(df, text_values, prefix=''):
    global embeddings_dim
    global embeddings
    global embeddings_path

    if not os.path.exists(uncompressed_embeddings_path):
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
        embeddings.save(uncompressed_embeddings_path)

    if not embeddings:
        embeddings = KeyedVectors.load(uncompressed_embeddings_path, mmap='r')

    print('Embeddings loaded!')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_values)
    inv_word_index = {v: k for k, v in tokenizer.word_index.items()}
    as_sequences = tokenizer.texts_to_sequences(text_values)
    vocab_size = len(tokenizer.word_index)
    percentage = len([1 for word in tokenizer.word_index if word in embeddings]) / vocab_size
    print(f"Percentage of words covered in the embeddings = {percentage}")

    matrix = np.zeros(shape=(len(as_sequences), embedding_dim))
    for j in range(len(as_sequences)):
        count = len(as_sequences[j])
        for k in range(len(as_sequences[j])):
            index = as_sequences[j][k]
            word = inv_word_index[index]
            if word in embeddings:
                matrix[j] += embeddings[word]
        if count > 0:
            matrix[j] /= count 
    
    column_names = [f'{prefix}text_embedding_{i}' for i in range(embedding_dim)]
    embeddings_df = pd.DataFrame(data=matrix, columns=column_names)
    embeddings_df['id'] = df['id']
    return pd.merge(embeddings_df, df, on='id')

def calculate_keyword_encoding(df, encoding_type='mean'):
    global mean_encodings

    df['keyword'] = df['keyword'].fillna('')
    df['keyword'] = df['keyword'].map(lambda x: _clean_keyword(x)) 

    if encoding_type == 'mean':
        if 'target' in df.columns:

            alpha = 5000.0
            global_mean = df['target'].mean()
            rows_range = len(df)
            
            df['mean_keyword'] = df.groupby('keyword')['target'].transform('mean')
            df['mean_encode'] = (rows_range * df['mean_keyword'] + global_mean * alpha)/(rows_range + alpha)
            mean_encodings = df.groupby('keyword')['mean_encode'].apply(lambda g: g.values[0]).to_dict()
            df.drop(['mean_keyword', 'mean_encode'], inplace=True, axis=1)

        df['mean_encode'] = df['keyword'].map(lambda x: mean_encodings[x])

    elif encoding_type == 'one_hot':
        unique_keywords = set(df['keyword'])
        for keyword in unique_keywords:
            df[keyword] = df['keyword'].map(lambda x: 1 if x == keyword else 0)

    elif encoding_type == 'mean_length':
        df['text_length'] = df['text'].map(lambda x: len(x))
        df['keywords_mean_length_encoding'] = df.groupby('keyword')['text_length'].transform('mean')
        df.drop(['text_length'], inplace=True, axis=1)

    elif encoding_type == 'binary':
        unique_keywords = set(df['keyword'])
        size = np.log2(len(unique_keywords)).round().astype(np.int8)

        def bin_array(num, m):
            return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

        i = 0
        map_keywords_binary = {}
        for keyword in unique_keywords:
            map_keywords_binary[keyword] = np.flip(bin_array(i,size))
            i += 1

        for column in range(size):
            df[f'c{column}'] = df['keyword'].map(lambda x: map_keywords_binary[x][column])
    elif 'none':
        pass

    else:
        raise KeyError(f'Invalid encoding {encoding_type}')


def add_manual_text_features(df):
    def _length(x):
        return len(x) if type(x) is str else 0

    try:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        _add_length_features(df)
        return

    df['keyword_length'] = df['keyword'].map(_length)
    df['text_length'] = df['text'].map(_length)
    df['location_length'] = df['location'].map(_length)

    df['stop_word_count'] = df['text'].map(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
    df['punctuation_count'] = df['text'].map(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['hashtag_count'] = df['text'].map(lambda x: x.count('#'))
    df['mention_count'] = df['text'].map(lambda x: x.count('@'))
    df['exclamation_count'] = df['text'].map(lambda x: x.count('!'))
    df['interrogation_count'] = df['text'].map(lambda x: x.count('?'))
    df['url_count'] = df['text'].map(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df['word_count'] = df['text'].map(lambda x: len(str(x).split()))
    df['unique_word_count'] = df['text'].map(lambda x: len(set(str(x).split())))
    df['space_in_keyword'] = df['keyword'].map(lambda x: x.count(' '))
    df['number_count'] = df['text'].map(lambda x: len([1 for y in x if y.isdigit()]))
    df['single_quote_count'] = df['text'].map(lambda x: x.count('\''))
    #df['asterisk_count'] = df['text'].map(lambda x: x.count('*'))
    #df['underscore_count'] = df['text'].map(lambda x: x.count('_'))
    #df['double_quote_count'] = df['text'].map(lambda x: x.count('\"'))
    df['single_quote_length'] = df['text'].map(lambda x: sum([len(x) for x in re.findall(r'\'.*?\'', x)]))
    #df['double_quote_length'] = df['text'].map(lambda x: sum([len(x) for x in re.findall(r'\".*?\"', x)]))
    #df['retweet_count'] = df['text'].map(lambda x: len(re.findall(r'\bRT\b', x.upper())) + len(re.findall(r'\bRETWEET\b', x.upper())))
    df['capitals_percentage'] = df['text'].map(
        lambda x: sum(1 for c in x if c.isupper() and c.isalpha()) / sum(1 for c in x if c.isalpha())
    )
    df['space_percentage'] = df['text'].map(lambda x: sum(1 for c in x if c.isspace()) / len(x))
    df['unique_chars'] = df['text'].map(lambda x: len(set(x)))
    df['unique_chars_percentage'] = df['unique_chars'] / df['text_length']
    #df['text_in_brackets'] = df['text'].map(lambda x: len(re.findall(r'\[.*?\]', x)))
    df['mention_chars'] = df['text'].map(lambda x: sum(len(x) for x in re.findall(r'@.*?\b', x)))
    df['mention_percentage'] = df['mention_chars'] / df['text_length']
    df['has_question_sign'] = df['text'].map(lambda x: (1 if (any(((c =='?') or (c =='¿')) for c in x)) else 0))
    df['has_uppercase'] = df['text'].map(lambda x: (1 if (any(c.isupper() for c in x)) else 0))

    time_pattern = r'\d:\d'
    df['time_in_text'] = df['text'].map(lambda x: len(re.findall(time_pattern, x)))
    emoji_pattern = r'(:|;)\s*?(\)|D|d|p|s|\/|\(|S|P)'
    df['emojis_in_text'] = df['text'].map(lambda x: len(re.findall(time_pattern, x)))

    df['word_density'] = df['word_count'] / (df['text_length'] + 1)
    df['capitals'] = df['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['num_unique_words'] = df['text'].apply(lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['word_count']


def add_location_features(df):
    def count_invalid_chars(x):
        if type(x) is float:
            return 0
        return len(re.findall(r'#|\$|\|%|\?|!|/|;|@|\+|\*|\d|:', x))

    df['location'] = df['location'].fillna('')
    
    df['invalid_location_character_count'] = df['location'].map(count_invalid_chars)
    df['location_is_place'] = df['location'].map(lambda x: len(re.findall(r'[a-z]*?,\s*[a-z]*', x.lower())))


def _clean_keyword(keyword):
    keyword = keyword.replace('%20', ' ')
    return keyword.lower()

def _clean_tweet(x):
    x = _decontracted(x)
    x = _remove_punctuations(x)
    x = word_tokenize(x)
    x = _remove_stopwords(x)
    x = _stemming_and_lemmatization(x)
    return ' '.join(x).lower()

def _decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def _stemming_and_lemmatization(x):
    lemmatizer = WordNetLemmatizer()
    y = [lemmatizer.lemmatize(w, get_pos(w)) for w in x]
    return y

def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def _remove_stopwords(x):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return ([word for word in x if word not in stopwords])

def _remove_punctuations(x):
    x = re.sub(re.compile('((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',flags=re.MULTILINE), '', x)
    x = re.sub('[^\w\s]','', x)
    x = re.sub("[^a-zA-Z\s]+", '', x)
    return x