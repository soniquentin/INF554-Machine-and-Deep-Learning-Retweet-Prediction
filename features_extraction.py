"""
    FUNCTIONS FOR FEATURES EXTRACTION
"""
import pandas as pd
from tqdm import tqdm

def nb_urls_hashtags(X, new_feature_calculated, debug = True):
    """
        Add the features [NUMBER OF TAGS] and [NUMBER OF HASHTAGS]
        Feature's name :
            - _#urls_
            - _#hashtags_

        UTILIZATION :
            X = nb_urls_hashtags(X)
    """
    if debug :
        print("    --> numberize_features ")

    if '_#urls_' not in X.columns or '_#hashtags_' not in X.columns :
        def str_to_len(x):
            if type(x) == str:
                return (len(x.split("'")) - 1)//2
            else:
                return x

        X["_#urls_"] = X["urls"].apply(str_to_len)
        X["_#hashtags_"] = X["hashtags"].apply(str_to_len)

        new_feature_calculated[0] |= True


    return X


def followers_over_friends(X, new_feature_calculated, debug = True) :
    """
        Add the feature [FOLLOWERS COUNT]/[FRIENDS COUNT]
        Feature's name :
            - _fof_
    """
    import numpy as np

    if debug :
        print("    --> followers_over_friends")

    if '_fof_' not in X.columns :
        X["_fof_"] = X["followers_count"]/(X["friends_count"] + 1) #+1 to avoid Nan and inf value

        #When friends_count = 0, fof = inf or Nan (depending of when followers_count > 0 or = 0). We have to treat these cases
        #X.loc[X["fof"] == np.inf , "fof"] = X["followers_count"]
        #X.loc[X["fof"].isna() , "fof"] = X["followers_count"]

        new_feature_calculated[0] |= True

    return X


def emotion_and_sentiments(X, new_feature_calculated, debug = True) :
    """
        Add a 10 dimensional vector to treat the emotion and sentiment in text using French-NRC-EmoLex.txt

        Features' names :
            - _anger_
            - _anticipation_
            - _disgust_
            - _fear_
            - _joy_
            - _negative_
            - _positive_
            - _sadness_
            - _surprise_
            - _trust_
    """

    if debug :
        print("    --> emotion_and_sentiments")

    emotions_features = ["_anger_",
                        "_anticipation_",
                        "_disgust_",
                        "_fear_",
                        "_joy_",
                        "_negative_",
                        "_positive_",
                        "_sadness_",
                        "_surprise_",
                        "_trust_"]

    if not( pd.Series(emotions_features).isin(X.columns).all() ) : #One emotion feature is missing

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

        new_feature_calculated[0] |= True


    return X
