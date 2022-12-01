from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

from os.path import exists
import os
import pickle
import csv


def numberize_features(X):
    """
        X_train = numberize_features(X_train)
        X_test = numberize_features(X_test)
    """
    def str_to_len(x):
        if type(x)==str:
            return len(x.split("'")) -1
        else:
            return x

    return X.applymap(str_to_len)


def import_model(model_name = "rf", debug = True) :
    """
        Import a model that already exist (.pickle)

        INPUT :
            model_name : model name (.pickle)
            debug : True if want to print for debug

        OUTPUT :
            rf

        EXAMPLE UTILIZATION :
            rf = import_model(model_name = "rf")
    """
    rf = None

    if debug :
        print("\n==== IMPORTING MODEL {} ====".format(model_name))

    file_exists = exists(os.path.dirname(__file__) + "/models/" + model_name)
    if file_exists :
        if debug :
            print("    --> Model found (Path : {})".format(os.path.dirname(__file__) + "/models/" + model_name))

        f = open(os.path.dirname(__file__) + "/models/" + model_name, 'rb')
        rf = pickle.load(f)
        f.close()
    else :
        if debug :
            print("    --> Model not found ! ")

    return rf


def import_features_data(data = "data/train.csv", debug = True) :
    """
        Import data and calculate features

        INPUT :
            data : file name of the raw data (.csv)
            debug : True if want to print for debug

        EXAMPLE UTILIZATION :
            df_data = import_features_data()
            X = df_data.drop(['retweets_count'], axis = 1, inplace = False )
            y = df_data["retweets_count"]
    """

    #========= IMPORT DATA ===========
    if debug :
        print("\n==== IMPORTING DATA (Path : {}) ====".format(data))
    df_data = pd.read_csv(data)


    #========= CALCULATE FEATURES ===========
    if debug :
        print("\n==== CALCULATING FEATURES ====")

    if debug :
        print("    --> Numberizing ")
    df_data = numberize_features(df_data)


    return df_data


def write_and_compare_prediction(rf, X, filename, compared_model, disagree_window = 0, debug = True) :
    """
        Create the file for submission and compare to a previous submission

        INPUT :
            rf : estimator
            X : Features of retweet to predict (should be from evaluation.csv)
            filename : the csv file name that will be created
            compared_model : the csv file name of the previous submission it will compare to
            disagree_window : show the disagree_window biggest absolute differences of prediction with the previous submissions
            debug : to print

        UTILIZATION :
            write_and_compare_prediction(rf, X, model_name, compared_model, debug)
    """
    if debug :
        print("\n==== WRITING CSV FOR SUBMISSION (File Name : {}) ====".format(filename))

    predictions = rf.predict(X).astype(int) ###TAKE THE PARTIE ENTIERE
    X["predictions"] = predictions


    with open('submissions/' + filename + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "retweets_count"])
        for index, row in X.iterrows():
            writer.writerow([ int(row["TweetID"]), int(row["predictions"])  ])


    if debug :
        print("\n==== COMPARING WITH PREVIOUS SUBMISSION (File Name : {}) ====".format(compared_model))

    previous_prediction = np.copy(predictions)

    with open('submissions/' + compared_model + ".csv", newline='') as g:
        reader = csv.reader(g, delimiter=',', quotechar='|')
        next(reader) #Skip first line

        index = 0
        for row in reader:
            previous_prediction[index] = int(row[1])
            index += 1

    print("    --> Mean square distance with a previous submission ({} VS {}) : {}".format(filename,
                                                                                            compared_model,
                                                                                            mean_squared_error(predictions, previous_prediction)))

    if disagree_window != 0 :
        diff = np.absolute(predictions - previous_prediction)
        ind = np.argpartition(diff, -disagree_window)[-disagree_window:] #Index of the disagree_window largest absolute differences
        print("    --> {} largest absolute differences".format(disagree_window) )
        for i in ind :
            print("        --> [TweetID : {}] Abs diff : {}      ( {} : {} ; {} : {} )".format( X._get_value(i, 'TweetID') ,
                                                                                diff[i] ,
                                                                                filename ,
                                                                                predictions[i] ,
                                                                                compared_model,
                                                                                previous_prediction[i] ))


def evaluation(rf, X_test, y_test):
    """
        evaluation(rf, X_test, y_test)
    """
    pred = rf.predict(X_test)
    for i in range(len(pred)):
        pred[i] = int(pred[i])
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
