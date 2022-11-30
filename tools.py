from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from os.path import exists
import os
import pickle


def numberize_features(X):
    """
        X_train = numberize_features(X_train)
        X_test = numberize_features(X_test)
    """

    for index, row in X.iterrows():

        for col in X.columns :
            if "str" in str(type(row[col])) :
                X.at[index, col] = row[col].count("'")

    return X


def import_data(filename = "evaluation_numberized", data = "data/train.csv", debug = True) :
    """
        Import data_numberized

        INPUT :
            filename : file name (.pickle) of the numberized data
            data : file name of the raw data (.csv)
            debug : True if want to print for debug

        EXAMPLE UTILIZATION :
            data_numberized = import_data()
            X = data_numberized.drop(['retweets_count'], axis = 1, inplace = False )
            y = data_numberized["retweets_count"]
    """

    if debug :
        print("\n==== CREATING NUMBERIZED DATA ====")

    file_exists = exists(os.path.dirname(__file__) + "/output/" + filename)
    if file_exists :
        if debug :
            print("    --> numberized data already exist (Path : {})".format(os.path.dirname(__file__) + "/output/" + filename))

        f = open(os.path.dirname(__file__) + "/output/" + filename, 'rb')
        data_numberized = pickle.load(f)
        f.close()
    else :
        if debug :
            print("    --> numberizing ... and saving at path {}...".format(os.path.dirname(__file__) + "/output/" + filename))

        data_numberized = pd.read_csv(data)
        data_numberized = numberize_features(data_numberized)

        with open(os.path.dirname(__file__) + "/output/" + filename, 'wb') as f :
            pickle.dump(data_numberized, f)

    return data_numberized


def evaluation(rf, X_test, y_test):
    """
        evaluation(rf, X_test, y_test)
    """
    pred = rf.predict(X_test)
    for i in range(len(pred)):
        pred[i] = round(pred[i])
    print("Mean square error : {}".format( mean_squared_error(y_test, pred)) )
    return mean_squared_error(y_test, pred)


def plot(X) :
    """
        plot(X_train)
    """
    X.drop(columns = ["text", "urls", "mentions", "hashtags", "TweetID"], inplace = True)

    #sns.distplot(X_train['retweets_count'])
    sns.pairplot(X.head(10000))
    plt.savefig('plot_result.png', dpi = 300)


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
