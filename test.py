from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import pad_sequences

from os.path import exists
import os
from tqdm import tqdm
import pickle
import numpy as np

from tools import *
import time

import matplotlib.pyplot as plt


def plot_color() :
    X = import_features_data()

    x = X["retweets_count"]
    y = X["fof"]

    info_dict = {}
    max_count = 0

    for i in range(len(x)) :
        if x[i] in info_dict :
            sum = info_dict[ x[i] ][0]
            count = info_dict[ x[i] ][1]
            info_dict[ x[i] ] = [sum + y[i] , count + 1]
            if count + 1 > max_count :
                max_count = count + 1
        else :
            info_dict[ x[i] ] = [ y[i] , 1]

    t_timeline = []
    y_timeline = []
    color_timeline = []

    for key in sorted(info_dict) :
        t_timeline.append(key)
        y_timeline.append(  info_dict[key][0]/info_dict[key][1]  )
        color_timeline.append( ( (info_dict[key][1]/max_count)**(1/7), 0,0) )

    plt.scatter(t_timeline, y_timeline, c = np.array(color_timeline),alpha=0.2 )
    plt.show()



def plot(X, debug = True) :
    """
        plot(X_train)
    """
    if debug :
        print("\n==== PLOTING (PAIRPLOT) ====")

    #X.drop(columns = ["text", "urls", "mentions", "hashtags", "TweetID"], inplace = True)
    X = X[['retweets_count', 'fof']]

    #sns.distplot(X_train['retweets_count'])
    sns.pairplot(X.head(10000))
    plt.savefig('output/plot_result.png', dpi = 300)


def sentiment_processing(X, columns_text_name) :
    """
        X_train = sentiment_processing(X_train, "text")
    """

    def get_subjectivity(text) :
        return TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0]

    def get_polarity(text) :
        return TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1]


    X["subjectivity"] = X[columns_text_name].apply(get_subjectivity) #Calcule la subjectivité
    X["polarity"] = X[columns_text_name].apply(get_polarity) #Calcule la polarité

    return X


def evaluation(rf, X_test, y_test):
    """
        evaluation(rf, X_test, y_test)
    """
    pred = rf.predict(X_test)
    for i in range(len(pred)):
        pred[i] = int(pred[i])
    print("Mean absolute error : {}".format( mean_absolute_error(y_test, pred)) )
    return mean_absolute_error(y_test, pred)



if __name__ == "__main__":

    X = pd.read_csv("data/train.csv")

    ti = time.time()
    def tokenize_hashtags(hashtags):
        hashtags = hashtags.replace('[', '')
        hashtags = hashtags.replace(']', '')
        hashtags = hashtags.replace("'", '')
        hashtags = hashtags.replace(" ", '')
        hashtags_list = hashtags.split(",")
        return hashtags_list

    X["hashtags_tokens"] = X["hashtags"].apply(tokenize_hashtags)
    vectorizer = TfidfVectorizer(analyzer = lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x for x in X["hashtags_tokens"]])
    vocab = list(vectorizer.get_feature_names())
    word_to_idx = {token:idx for idx, token in enumerate(vocab) if token != ''}


    # converting the docs to their token ids
    def indexilze(hashtags_tokens) :
        list_to_return = []
        for token in hashtags_tokens :
            try :
                list_to_return.append(word_to_idx[token])
            except KeyError : #the tokens is not present enough (min_df) in the database to be considered as feeature
                pass
        return list_to_return

    X["hashtags_tokens"] = X["hashtags_tokens"].apply(indexilze)

    one_hot = np.zeros((len(X.index), len(vocab)))
    index = 0

    for x in X["hashtags_tokens"] :
        for token_index in x :
            one_hot[index, token_index] = 1
        index += 1

    embed = pd.DataFrame(one_hot, columns = vocab)

    """
    token_ids_padded = pad_sequences(token_ids, padding="post")
    token_ids = token_ids.reshape(-1, 1)
    # convert the token ids to one hot representation
    one_hot = OneHotEncoder()
    X = one_hot.fit_transform(token_ids_padded)
    # converting to dataframe
    X_df = pd.DataFrame(X.toarray())

    X_df
    """
