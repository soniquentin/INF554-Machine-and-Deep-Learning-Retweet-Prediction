"""
    FUNCTIONS FOR FEATURES EXTRACTION
"""
import pandas as pd
from tqdm import tqdm
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import os
import pickle

def nb_urls_hashtags(X, debug = True):
    """
        Add the features [NUMBER OF TAGS] and [NUMBER OF HASHTAGS]
        Feature's name :
            - #urls
            - #hashtags

        UTILIZATION :
            X = nb_urls_hashtags(X)
    """
    if debug :
        print("    --> numberize_features ")


    def str_to_len(x):
        if type(x) == str:
            return (len(x.split("'")) - 1)//2
        else:
            return x

    X["#urls"] = X["urls"].apply(str_to_len)
    X["#hashtags"] = X["hashtags"].apply(str_to_len)

    return X


def followers_and_friends(X, debug = True) :
    """
        Add the feature [FOLLOWERS COUNT]/[FRIENDS COUNT], [FOLLOWERS COUNT]*[FAVORITES COUNT], [FRIENDS COUNT]*[FAVORITES COUNT]
        Feature's name :
            - fof
            - ftf1
            - ftf2
    """
    import numpy as np

    if debug :
        print("    --> followers_and_friends")


    X["fof"] = X["followers_count"]/(X["friends_count"] + 1) #+1 to avoid Nan and inf value
    X["ftf1"] = X["followers_count"]*X["favorites_count"]
    X["ftf2"] = X["friends_count"]*X["favorites_count"]


    #When friends_count = 0, fof = inf or Nan (depending of when followers_count > 0 or = 0). We have to treat these cases
    #X.loc[X["fof"] == np.inf , "fof"] = X["followers_count"]
    #X.loc[X["fof"].isna() , "fof"] = X["followers_count"]


    return X


def timestamp_features(X, debug = True) :
    """
        Make an hour feature (periodical)
        Feature's name :
            - hour
            - weekday
    """
    if debug :
        print("    --> timestamp_features")

    def t_hour(timestamp):
        T = 24*3600 #Periode
        T_0 = 1646866800 #Origine time (correspond to a date at midnight)
        timestamp //= 1000 #Timestamp as 000 at the end
        return np.sin(   2*np.pi*(timestamp - T_0)/T  )

    def t_weekday(timestamp):
        T = 24*3600*7 #Periode
        T_0 = 1646866800 #Origine time (correspond to a date at midnight)
        timestamp //= 1000 #Timestamp as 000 at the end
        return np.sin(   2*np.pi*(timestamp - T_0)/T  )

    X["hour"] = X["timestamp"].apply(t_hour)
    X["weekday"] = X["timestamp"].apply(t_weekday)

    return X



def emotion_and_sentiments(X, debug = True) :
    """
        Add a 10 dimensional vector to treat the emotion and sentiment in text using French-NRC-EmoLex.txt

        Features' names :
            - anger
            - anticipation
            - disgust
            - fear
            - joy
            - negative
            - positive
            - sadness
            - surprise
            - trust
    """

    if debug :
        print("    --> emotion_and_sentiments")

    emotions_features = ["anger",
                        "anticipation",
                        "disgust",
                        "fear",
                        "joy",
                        "negative",
                        "positive",
                        "sadness",
                        "surprise",
                        "trust"]

    df = pd.read_csv("data\French-NRC-EmoLex.txt", delimiter = "\t")

    #Initialize at a empty columns
    for feature in emotions_features :
        X[feature] = ""

    for index, row in tqdm(X.iterrows()):

        tokens = row["text"].split(" ") #Split to get list of words
        sub_df = df[df['French Word'].isin(tokens)].drop( ['English Word' , 'French Word'], axis = 1, inplace = False ) #Panda dataframe with emotion (columns) and words in text (rows)
        sub_df_sum = sub_df.sum() #Sum to count how many words
        sub_df_norm = sub_df.div( sub_df.sum(axis=1)  , axis=0) #Normalize each row by their sum to get kind of probabilities for each word. When sum is 0, replace by Nan
        sub_df_norm = sub_df_norm.fillna(0)
        sub_df_norm_sum = sub_df_norm.sum() #Sum all the probabilities on all words

        sub_df_sum *= sub_df_norm_sum


        sub_df_sum /= max(1, sub_df_sum.max()) #Normalize

        #Updates values
        for feature in emotions_features :
            X.at[index, feature] = sub_df_sum[feature.replace('_', '')]


    return X



def text_embedding(X, model_name, word2vec_model, tfidf, vocab_hashtag, n_dim = 100, debug = True) :
    """
        Transform the text of each tweet into a n_dim vector

        Features' names :
            - text_emb_i (0 <= i < n_dim)

        INPUT :
            - X : dataframe of data
            - model_name : ML model name that will fit or predict after
            - word2vec_model : word2vec model that will be used to embed the tweets
            - tfidf : tfidf score from the training phase
            - n_dim : dimension of vector reprenting the sentences
            - debug : to print
    """

    if debug :
        print("    --> text_embedding")
        print("        - tokenizing tweets")

    def tokenize(tweet):
        tokens = tweet.split(" ")
        return tokens
    X["text_tokens"] = X["text"].apply(tokenize)

    def tokenize_hashtags(hashtags):
        hashtags = hashtags.replace('[', '')
        hashtags = hashtags.replace(']', '')
        hashtags = hashtags.replace("'", '')
        hashtags = hashtags.replace(" ", '')
        hashtags_list = hashtags.split(",")
        return hashtags_list

    X["hashtags_tokens"] = X["hashtags"].apply(tokenize_hashtags)



    if word2vec_model == None or tfidf == None or vocab_hashtag == None : #Training phase
        if debug :
            print("        - training a Word2Vec")

        word2vec_model = Word2Vec(window = 5, min_count = 2, workers = 4, vector_size = n_dim)
        word2vec_model.build_vocab(X["text_tokens"], progress_per = 10000)
        word2vec_model.train(X["text_tokens"], total_examples = word2vec_model.corpus_count, epochs = word2vec_model.epochs)
        word2vec_model.save("./models/word2vec_" +  model_name + ".model") #Save the word2vec model



        if debug :
            print("        - calculating the TF-IDF score for each words/tokens")

        vectorizer = TfidfVectorizer(analyzer = lambda x: x, min_df=10)
        matrix = vectorizer.fit_transform([x for x in X["text_tokens"]])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        with open(os.path.dirname(__file__) + "/models/tfidf_" + model_name, 'wb') as f : #save the tidtf
            pickle.dump(tfidf, f)


        if debug :
            print("        - creating embedding input with hashtags")

        vectorizer = TfidfVectorizer(analyzer = lambda x: x, min_df=15)
        matrix = vectorizer.fit_transform([x for x in X["hashtags_tokens"]])
        vocab_hashtag = list(vectorizer.get_feature_names())
        with open(os.path.dirname(__file__) + "/models/vocab_hashtag_" + model_name, 'wb') as f : #save the vocab_hashtag
            pickle.dump(vocab_hashtag, f)


    if debug :
        print("        - creating the final vector that represents each tweet")

    def buildWordVector(tokens):
        vec = np.zeros(n_dim).reshape((1, n_dim))
        count = 0
        for word in tokens:
            try:
                vec += word2vec_model.wv[word]*tfidf[word]
                count += tfidf[word]
            except KeyError: # handling the case where the token is not
                             # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec

    X = pd.concat( [X,
                    pd.DataFrame(np.concatenate([buildWordVector(x) for x in X["text_tokens"]]), columns = ["text_emb_{}".format(i) for i in range(n_dim)])] ,
                    axis = 1)
    X.drop(["text_tokens"], axis = 1, inplace = True)



    if debug :
        print("        - creating the final one-hot-encoder that represents each hashtags")

    word_to_idx = {token:idx + 1 for idx, token in enumerate(vocab_hashtag) if token != ''}
    # converting the docs to their token ids
    input_length = 4 #input_length for embed
    def indexilze(hashtags_tokens) :
        list_to_return = [0]*input_length
        index_list = 0
        for token in hashtags_tokens :
            try :
                list_to_return[index_list] = word_to_idx[token]
                index_list += 1
            except Exception as e : #the tokens is not present enough (min_df) in the database to be considered as feeature OR index out of range
                pass
        return list_to_return
    X["hashtags_tokens"] = X["hashtags_tokens"].apply(indexilze)
    """
    one_hot = np.zeros((len(X.index), len(vocab_hashtag)))
    index = 0
    for x in X["hashtags_tokens"] :
        for token_index in x :
            one_hot[index, token_index] = 1
        index += 1s
    embed = pd.DataFrame(one_hot, columns = vocab_hashtag) #The one-hot-encoder
    X.drop(["hashtags_tokens"], axis = 1, inplace = True)
    """
    embed = pd.DataFrame()
    for i in range(input_length) :
        embed["{}_ht".format(i)] = pd.Series([ x[i] for x in X["hashtags_tokens"] ])

    X.drop(["hashtags_tokens"], axis = 1, inplace = True)


    return X, [embed, len(vocab_hashtag) + 1]
