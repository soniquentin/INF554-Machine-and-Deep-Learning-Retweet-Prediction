from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

from os.path import exists
import os
from tqdm import tqdm
import pickle
import numpy as np

from tools import *

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



if __name__ == "__main__":

    X = pd.read_csv("data/train.csv")

    def str_to_len(x):
        if type(x) == str:
            return (len(x.split("'")) - 1)//2
        else:
            return x



    X["nb_urls"] = X["urls"].apply(str_to_len)
    X["nb_hashtags"] = X["hashtags"].apply(str_to_len)

    print(X)

    X.to_csv("data/train2.csv", sep=',', encoding='utf-8', index = False)
