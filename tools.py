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
from models import *
from sklearn.preprocessing import StandardScaler
from gensim.models.word2vec import Word2Vec

from features_extraction import *

from os.path import exists
import os
import pickle
import csv



def import_model(model_name = "rf", debug = True) :
    """
        Import a model that already exist (.pickle) and all the training objects (scaler, word2vec, TFIDF)

        INPUT :
            model_name : model name (.pickle)
            debug : True if want to print for debug

        OUTPUT :
            - rf
            - training_objs

        EXAMPLE UTILIZATION :
            rf,training_objs = import_model(model_name = "rf")
    """
    rf = None

    if debug :
        print("\n==== IMPORTING MODEL {} ====".format(model_name))

    training_objs = {}



    ### IMPORT MODEL
    file_exists = exists(os.path.dirname(__file__) + "/models/" + model_name)
    if file_exists :
        if debug :
            print("    --> Model found (Path : {})".format(os.path.dirname(__file__) + "/models/" + model_name))
        try :
            rf = load_model(os.path.dirname(__file__) + "/models/" + model_name + ".h5")
        except Exception as e: #Not NN
            f = open(os.path.dirname(__file__) + "/models/" + model_name, 'rb')
            rf = pickle.load(f)
            f.close()
    else :
        if debug :
            print("    --> Model not found ! ")






    ## IMPORT SCALER
    file_exists = exists(os.path.dirname(__file__) + "/models/" + "scaler_" + model_name)
    if file_exists :
        if debug :
            print("    --> Scaler found (Path : {})".format(os.path.dirname(__file__) + "/models/" + "scaler_" + model_name))

        f = open(os.path.dirname(__file__) + "/models/scaler_" + model_name, 'rb')
        training_objs["scaler"] = pickle.load(f)
        f.close()
    else :
        if debug :
            print("    --> Scaler not found ! ")





    ## IMPORT WORD2VEC
    file_exists = exists(os.path.dirname(__file__) + "/models/word2vec_" + model_name + ".model")
    if file_exists :
        if debug :
            print("    --> Word2vec found (Path : {})".format(os.path.dirname(__file__) + "/models/word2vec_" + model_name + ".model"))
        training_objs["word2vec"] = Word2Vec.load(os.path.dirname(__file__) + "/models/word2vec_" + model_name + ".model")
    else :
        if debug :
            print("    --> Word2vec not found ! ")




    ## IMPORT TFIDF
    file_exists = exists(os.path.dirname(__file__) + "/models/tfidf_" + model_name)
    if file_exists :
        if debug :
            print("    --> TFIDF found (Path : {})".format(os.path.dirname(__file__) + "/models/tfidf_" + model_name))

        f = open(os.path.dirname(__file__) + "/models/tfidf_" + model_name, 'rb')
        training_objs["tfidf"] = pickle.load(f)
        f.close()
    else :
        if debug :
            print("    --> TFIDF not found ! ")



    ## IMPORT VOCAB HASHTAG
    file_exists = exists(os.path.dirname(__file__) + "/models/vocab_hashtag_" + model_name)
    if file_exists :
        if debug :
            print("    --> vocab_hashtag found (Path : {})".format(os.path.dirname(__file__) + "/models/vocab_hashtag_" + model_name))

        f = open(os.path.dirname(__file__) + "/models/vocab_hashtag_" + model_name, 'rb')
        training_objs["vocab_hashtag"] = pickle.load(f)
        f.close()
    else :
        if debug :
            print("    --> vocab_hashtag not found ! ")





    if debug and rf != None :
        try :
            print("    --> Model parameters : {}".format( rf.get_params() ) )
        except Exception as e :
            pass #rf doesn't have get_params() if it is a neural network

    return rf, training_objs


def import_features_data(data = "data/train.csv",
                        feat_drop = ['text', 'mentions', 'urls', 'hashtags', 'TweetID'], #+ ["fof", "ftf1", "ftf2", "favorites_count", "followers_count", "statuses_count", "friends_count"],
                        feat_scale = [], #["fof", "favorites_count", "followers_count", "statuses_count", "friends_count", "timestamp"],
                        feat_log = [],#["fof", "ftf1", "ftf2", "favorites_count", "followers_count", "statuses_count", "friends_count"],
                        n_dim = 10,
                        saved = False,
                        training_objs = None,
                        model_name = "rf2",
                        debug = True):
    """
        Import data and calculate features

        INPUT :
            data : path to csv of the raw data
            debug : True if want to print for debug
            feat_drop : features to drop at the end (that will not be trained)
            feat_scale : features to be scaled
            feat_log : features that to have to log_transformed
            n_dim : dimension for text text_embedding
            saved : if the features have to be saved (overwrite data !!!)
            training_objs : dictionnary of object from training. (Useful during testing). During training, training_objs = None
            model_name : ML model name (if training, that will be trained. if testing, the one that will predict)

        EXAMPLE UTILIZATION :
            df_data = import_features_data()
            X = df_data.drop(['retweets_count'], axis = 1, inplace = False )
            y = df_data["retweets_count"]
    """
    if training_objs != None : #Testing phase ==> reintegrate training objects
        if "scaler" in training_objs :
            scaler = training_objs["scaler"]
        if "word2vec" in training_objs :
            word2vec_model = training_objs["word2vec"]
        if "tfidf" in training_objs :
            tfidf = training_objs["tfidf"]
        if "vocab_hashtag" in training_objs :
            vocab_hashtag = training_objs["vocab_hashtag"]
    else :  #Training phase ==> we have to build training objects
        scaler = None
        word2vec_model = None
        tfidf = None
        vocab_hashtag = None

    embed = None

    #========= IMPORT DATA ===========
    if debug :
        print("\n==== IMPORTING DATA (Path : {}) ====".format(data))
    df_data = pd.read_csv(data)


    #========= EXTRACTING FEATURES ===========
    if debug :
        print("\n==== EXTRACTING FEATURES ====")

    df_data = nb_urls_hashtags(df_data, debug) #Number urls and number hashtags
    df_data = timestamp_features(df_data, debug)
    #df_data = emotion_and_sentiments(df_data, debug) #Add sentiment features
    df_data = followers_and_friends(df_data, debug) #followers_count/friends_count
    df_data, embed = text_embedding(df_data, model_name, word2vec_model, tfidf, vocab_hashtag, n_dim = n_dim, debug = True)

    if saved: #A new feature was calculated, we have to update the features in the file
        if debug :
            print("    --> Updating {} with new features".format(data))
        df_data.to_csv(data, sep=',', encoding='utf-8', index = False) #Saving calculated features to avoid recalculate them again


    if debug :
        print("    --> Log-transforming and scaling features...")

    #log-transform
    def log_transform(x) :
        return np.log(x + 1)
    for feature in feat_log :
        df_data["{}_log".format(feature)] = df_data[feature].apply(log_transform)

    #scaling
    """
    emb_text_feat = ["text_emb_{}".format(i) for i in range(n_dim)]
    scaled_sub_df = df_data[feat_scale + emb_text_feat] #We add text_emb_i features to be scaled
    df_data.drop(emb_text_feat, axis = 1, inplace = True) #We drop text_emb_i on the df_data
    scaled_sub_df.rename(columns = {feature : "{}_scaled".format(feature) for feature in feat_scale}, inplace = True) #Rename the features except the text embedding one

    if training_objs != None : #Testing phase
        #Scale with the scaler coming from the training phase
        scaled_sub_df = pd.DataFrame(scaler.transform(scaled_sub_df), index=scaled_sub_df.index, columns=scaled_sub_df.columns)
    else : #Training phase
        scaler = StandardScaler() #create the scaler
        scaled_sub_df = pd.DataFrame(scaler.fit_transform(scaled_sub_df), index=scaled_sub_df.index, columns=scaled_sub_df.columns)
        #Save the scaler
        with open(os.path.dirname(__file__) + "/models/scaler_" + model_name, 'wb') as f :
            pickle.dump(scaler, f)

    df_data = pd.concat([df_data, scaled_sub_df], axis = 1) #Remerge the two
    """




    #========= SELECTING FEATURES ===========
    if debug :
        print("\n==== SELECTING FEATURES ====")
        print("    --> Features dropped :", feat_drop)

    tweed_id = df_data["TweetID"]
    df_data.drop(feat_drop, axis = 1, inplace = True )

    if debug :
        print("    --> Final Features :", df_data.columns)


    if training_objs == None : #Training phase
        #y = df_data["retweets_count"].apply(log_transform) #we log transform the target as well
        y = df_data["retweets_count"]
        df_data.drop(['retweets_count'], axis = 1, inplace = True )
    else : #Testing phase
        y = None


    #pd.set_option('display.max_columns', 500)
    #print(df_data.head(10))
    #print(y.head(10))
    return df_data, embed, y, tweed_id



def train_model(alg = "RF", data = "data/train.csv", save_model = True, model_name = "rf2", debug = True  , **kwargs) : #kwargs is a dictionnary
    """
        Import train data, train an estimator with **kwargs as arguments and save it

        INPUT :
            alg : the choosen algorithm ("RF" : Random Forst // "GB" : Gradient Boosting // "XGB" : XGBoost)
            data : the path with train data
            save_model : True if you want to save the model
            model_name : model name file (.pickle) that will be saved
            debug : to print

        OUTPUT : the scaler that was used to standardize the data
    """

    X, embed, y , tweed_id = import_features_data(data = data, model_name = model_name, debug = debug)

    ### Overwrite Warning msg
    file_exists = exists(os.path.dirname(__file__) + "/models/" + model_name)
    if file_exists :
        print("WARNING : model {} already exists. Interrupt now, otherwise, it will be overwritten".format(model_name))


    ### TRAINING ###
    if debug :
        print("\n==== TRAINING... ====")
        t_i = time.time()


    if "NN" not in alg : #alg is not a neural network
        if alg == "RF" :
            rf = RandomForestRegressor(**kwargs)
        elif alg == "GB" :
            rf = GradientBoostingRegressor(**kwargs)
        else :
            rf = xgb.XGBRegressor(**kwargs)

        rf.fit(X, y)

    else : #Use neural network
        model = Sequential()
        rf = deepnet2(len(X.columns), embed[1])

        if debug :
            plot_model(rf, to_file='./models/{}.png'.format(model_name))
            rf.summary()
        print(type(X))
        print(type(embed[0]))
        print(type(y))
        rf.fit([X,embed[0]],y, **kwargs)



    if debug :
        print("    --> training duration : {}".format(time.time() - t_i))

    ### SAVING MODEL ###
    if save_model :
        if debug :
            print("\n==== SAVING MODEL... ====")
        if "NN" not in alg :
            with open(os.path.dirname(__file__) + "/models/" + model_name, 'wb') as f :
                pickle.dump(rf, f)
        else :
            rf.save(os.path.dirname(__file__) + "/models/" + model_name + ".h5")




def write_and_compare_prediction(rf, X, embed, tweed_id, filename, compared_model, disagree_window = 0, debug = True) :
    """
        Create the file for submission and compare to a previous submission

        INPUT :
            rf : estimator
            X : Features of retweet to predict (should be from evaluation.csv)
            tweed_id : to get back the tweet id columns
            filename : the csv file name that will be created
            compared_model : the csv file name of the previous submission it will compare to
            disagree_window : show the disagree_window biggest absolute differences of prediction with the previous submissions
            debug : to print

        UTILIZATION :
            write_and_compare_prediction(rf, X, model_name, compared_model, debug)
    """
    if debug :
        print("\n==== WRITING CSV FOR SUBMISSION {} ====".format(filename))


    try :
        predictions = rf.predict([X,embed[0]])
        predictions = predictions.reshape(len(predictions))
        def log_transform_inverse(x) :
            return round(np.exp(x) - 1)
        np.vectorize(log_transform_inverse)
        predictions = log_transform_inverse(predictions)

    except Exception as e : #Error if not a NN
        predictions = rf.predict(X).astype(int)

    X["predictions"] = predictions
    X["TweetID"] = tweed_id




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
