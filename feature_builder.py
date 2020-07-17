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

embedding_dim = 300 
embeddings_path = './data/embeddings/word2vec.bin'
embeddings = None
mean_encodings = None

def process_dataset(df, encoding_type='mean', text_type='embeddings', target_dimensions=None, clean_text=False):
    df2 = df.copy()
    global feature_names

    add_location_features(df2)
    calculate_keyword_encoding(df2, encoding_type=encoding_type)
    add_manual_text_features(df2)

    if text_type == 'embeddings':
        add_text_embeddings(df2, clean_text=clean_text)

    elif text_type == 'tfidf':
        add_text_tfidf(df2, clean_text=clean_text)

    elif text_type == 'bow':
        add_text_bow(df2, clean_text=clean_text)

    elif text_type == 'none':
        pass
    
    df2.drop(['text', 'location', 'keyword', 'id'], axis=1, inplace=True)
    if 'target' in df2.columns:
        df2.drop(['target'], axis=1, inplace=True)

    if target_dimensions:
        df2 = reduce_dimensions(df2, dims=target_dimensions)

    return df2


def _add_text_using_vectorizer(df, clean_text, vectorizer):
    text_values = [ _clean_tweet(x) if clean_text else x for x in df['text'].values]
    
    matrix = vectorizer.fit_transform(text_values)
    feature_names = vectorizer.get_feature_names()

    results_df = pd.DataFrame(matrix.T.todense(), index=feature_names, columns=[n for n in range(len(text_values))])
    for i in range(len(feature_names)):
        feature = feature_names[i]
        feature_column = results_df.loc[feature, :]
        if feature_column.sum() >= 0.5:
            df[feature] = results_df.loc[feature, :]



def add_text_bow(df, clean_text):
    _add_text_using_vectorizer(df, clean_text, CountVectorizer())
    

def add_text_tfidf(df, clean_text):
    _add_text_using_vectorizer(df, clean_text, TfidfVectorizer())



def add_text_embeddings(df):
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
    vocab_size = len(tokenizer.word_index)
    percentage = len([1 for word in tokenizer.word_index if word in embeddings]) / vocab_size
    print(f"Percentage of words covered in the embeddings = {percentage}")

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

def calculate_mean_encoding(df, encoding_type='mean'):
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

    elif 'one_hot':
        unique_keywords = set(df['keyword'])
        for keyword in unique_keywords:
            df[keyword] = df['keyword'].map(lambda x: 1 if x == keyword else 0)

    elif 'mean_length':
        df['text_length'] = df['text'].map(lambda x: len(x))
        df['keywords_mean_length_encoding'] = df.groupby('keyword')['text_length'].transform('mean')
        df.drop(['text_length'], inplace=True, axis=1)

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

    _calculate_mean_encoding(df)

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