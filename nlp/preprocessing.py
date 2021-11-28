import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from collections import Counter
import nlp.utils as utils
import nlp.slang as slang

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))

def validateInput(series):
    input_datatype = type(series)
    if(pd.Series != input_datatype):
         raise Exception("input should be pandas Series")

def to_lower(series):
    validateInput(series)
    return series.str.lower()

def remove_punctuation(series):
    validateInput(series)
    return series.apply(lambda text: utils.remove_punctuation(text,PUNCT_TO_REMOVE))

def remove_stopwords(series):
    validateInput(series)
    return series.apply(lambda text: utils.remove_stopwords(text,STOPWORDS))

def remove_freqwords(series):
    validateInput(series)
    cnt = Counter()

    for text in series.values:
        for word in text.split():
            cnt[word] += 1

    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    return series.apply(lambda text: utils.remove_freqwords(text,FREQWORDS))

def remove_rarewords(series,n_rare_words=20):
    validateInput(series)
    cnt = Counter()

    for text in series.values:
        for word in text.split():
            cnt[word] += 1

    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    return series.apply(lambda text: utils.remove_rarewords(text,RAREWORDS))

def stem_words(series):
    validateInput(series)
    return series.apply(lambda text: utils.stem_words(text))

def lemmatize_words_without_pos(series):
    return series.apply(lambda text: utils.lemmatize_words_without_pos(text))

def lemmatize_words(series):
    validateInput(series)
    return series.apply(lambda text: utils.lemmatize_words(text))

def remove_emoji(series):
    validateInput(series)
    return series.apply(lambda text: utils.remove_emoji(text))

def remove_urls(series):
    validateInput(series)
    return series.apply(lambda text: utils.remove_urls(text))

def remove_html(series):
    validateInput(series)
    return series.apply(lambda text: utils.remove_html(text))

def remove_slang_words(series):
    validateInput(series)

    chat_words_map_dict = {}
    chat_words_list = []

    for line in slang.slang_words_str.split("\n"):
        if line != "":
            words = line.split('=')
            if(len(words) == 2):
                cw = words[0]
                cw_expanded = words[1]
                chat_words_list.append(cw)
                chat_words_map_dict[cw] = cw_expanded
    chat_words_list = set(chat_words_list)
    return series.apply(lambda text: utils.remove_slang_words(text,chat_words_list,chat_words_map_dict))

        
    
