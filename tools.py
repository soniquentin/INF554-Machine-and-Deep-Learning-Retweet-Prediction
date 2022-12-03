from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import time
import xgboost as xgb

from features_extraction import *

from os.path import exists
import os
import pickle
import csv


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

    if debug and rf != None :
        print("    --> Model parameters : {}".format( rf.get_params() ) )

    return rf


def import_features_data(data = "data/train.csv", list_features_to_drop = ['text', 'mentions', 'urls', 'hashtags'], debug = True) :
    """
        Import data and calculate features

        INPUT :
            data : path to csv of the raw data
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


    #========= EXTRACTING FEATURES ===========
    if debug :
        print("\n==== EXTRACTING FEATURES ====")

    new_feature_calculated = [False] ##Variable that is True if at least one feature have to calculated

    df_data = nb_urls_hashtags(df_data, new_feature_calculated, debug) #Number urls and number hashtags
    df_data = emotion_and_sentiments(df_data, new_feature_calculated, debug) #Add sentiment features
    df_data = followers_over_friends(df_data,new_feature_calculated, debug) #followers_count/friends_count

    if new_feature_calculated : #A new feature was calculated, we have to update the features in the file
        if debug :
            print("    --> Updating {} with new features".format(data))
        df_data.to_csv(data, sep=',', encoding='utf-8', index = False) #Saving calculated features to avoid recalculate them again

    #========= SELECTING FEATURES ===========
    if debug :
        print("\n==== SELECTING FEATURES ====")
        print("    --> Features dropped :", list_features_to_drop)

    df_data.drop(list_features_to_drop, axis = 1, inplace = True )

    return df_data



def train_model(alg = "RF", data = "data/train.csv", save_model = True, model_name = "rf2", debug = True  , **kwargs) : #kwargs is a dictionnary
    """
        Import train data, train an estimator with **kwargs as arguments and save it

        INPUT :
            alg : the choosen algorithm ("RF" : Random Forst // "GB" : Gradient Boosting // "XGB" : XGBoost)
            data : the path with train data
            save_model : True if you want to save the model
            model_name : model name file (.pickle) that will be saved
            debug : to print
    """

    df_data = import_features_data(data = data, debug = debug)

    try :
        X = df_data.drop(['retweets_count'], axis = 1, inplace = False )
        y = df_data["retweets_count"]
    except Exception as e :
        raise Exception("Are you sure that file {} contains the column 'retweets_count' ?".format(data))

    ### Overwrite Warning msg
    file_exists = exists(os.path.dirname(__file__) + "/models/" + model_name)
    if file_exists :
        print("WARNING : model {} already exists. Interrupt now, otherwise, it will be overwritten".format(model_name))


    ### TRAINING ###
    if debug :
        print("\n==== TRAINING... ====")
        t_i = time.time()

    if alg == "RF" :
        rf = RandomForestRegressor(**kwargs)
    elif alg == "GB" :
        rf = GradientBoostingRegressor(**kwargs)
    else :
        rf = xgb.XGBRegressor(**kwargs)


    rf.fit(X, y)
    if debug :
        print("    --> training duration : {}".format(time.time() - t_i))

    ### SAVING MODEL ###
    if save_model :
        if debug :
            print("\n==== SAVING MODEL... ====")
        with open(os.path.dirname(__file__) + "/models/" + model_name, 'wb') as f :
            pickle.dump(rf, f)



def semi_supervised(alg = "RF", previous_model_name = "rf2", save_model = True, data_train = "data/train.csv", data_test = "data/evaluation.csv", debug = True, **kwargs):
    """
        Once a first training was done, process to a semi_supervised approach :
            - Labelling the evaluation set with a previous_model to obtain a bigger training
            - train an new estimator with **kwargs as arguments and save it
        The new estimator will have the same name as the previous one with "_semisupervised" appended

        INPUT :
            - alg : the choosen algorithm for the new estimator ("RF" : Random Forst // "GB" : Gradient Boosting // "XGB" : XGBoost)
            - previous_model_name : name of the previous model (that is used for labelling evaluation)
            - save_model : True if you want to save the new model
            - data_train : path to csv of the training data
            - data_train : path to csv of the evaluation data
            - debug : to print
    """

    rf = import_model(model_name = previous_model_name)

    ### Overwrite Warning msg
    file_exists = exists(os.path.dirname(__file__) + "/models/" + previous_model_name + "_semisupervised")
    if file_exists :
        print("WARNING : model {} already exists. Interrupt now, otherwise, it will be overwritten".format(previous_model_name + "_semisupervised"))


    ### TRAINING ###
    if debug :
        print("\n==== TRAINING SEMI-SURPERVISED ====")
        t_i = time.time()
        print("    --> Expanding the training dataset ... (labelling evaluation data)")

    #Import train and evaluation abd label evaluation
    df_data_train = import_features_data(data = data_train, list_features_to_drop = ['text', 'mentions', 'urls', 'hashtags'], debug = True)

    df_data_test = import_features_data(data = data_test, list_features_to_drop = ['text', 'mentions', 'urls', 'hashtags'], debug = True)
    df_data_test["retweets_count"] = rf.predict(df_data_test).astype(int)

    #Create new dataset by concatinating train et evaluation set
    df_data = pd.concat([df_data_train, df_data_train])
    X = df_data.drop(['retweets_count'], axis = 1, inplace = False )
    y = df_data["retweets_count"]

    if alg == "RF" :
        rf = RandomForestRegressor(**kwargs)
    elif alg == "GB" :
        rf = GradientBoostingRegressor(**kwargs)
    else :
        rf = xgb.XGBRegressor(**kwargs)

    if debug :
        print("    --> Fitting...")

    rf.fit(X, y)
    if debug :
        print("    --> training duration : {}".format(time.time() - t_i))

    ### SAVING MODEL ###
    if save_model :
        if debug :
            print("\n==== SAVING MODEL... ====")
        with open(os.path.dirname(__file__) + "/models/" + previous_model_name  + "_semisupervised", 'wb') as f :
            pickle.dump(rf, f)




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
        print("\n==== WRITING CSV FOR SUBMISSION {} ====".format(filename))

    predictions = rf.predict(X).astype(int) ###TAKE THE PARTIE ENTIERE
    X["predictions"] = predictions


    with open('submissions/' + filename + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "retweets_count"])
        for index, row in X.iterrows():
            writer.writerow([ int(row["TweetID"]), int(row["predictions"])  ])


    if debug :
        print("\n==== COMPARING WITH PREVIOUS SUBMISSION {} ====".format(compared_model))

    previous_prediction = np.copy(predictions)

    with open('submissions/' + compared_model + ".csv", newline='') as g:
        reader = csv.reader(g, delimiter=',', quotechar='|')
        next(reader) #Skip first line

        index = 0
        for row in reader:
            previous_prediction[index] = int(row[1])
            index += 1

    print("    --> Mean absolute distance with a previous submission ({} VS {}) : {}".format(filename,
                                                                                            compared_model,
                                                                                            mean_absolute_error(predictions, previous_prediction)))

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
    print("Mean absolute error : {}".format( mean_absolute_error(y_test, pred)) )
    return mean_absolute_error(y_test, pred)



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
