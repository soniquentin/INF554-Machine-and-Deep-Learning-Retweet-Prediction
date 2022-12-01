from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from os.path import exists
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt


if __name__ == "__main__":
    df_data = pd.read_csv("data/train.csv")
    df_data["New"] = df_data["followers_count"]/df_data["friends_count"]
    #df_data.replace([np.inf], np.nan , inplace=True)
    #df_data.replace([np.nan], df_data["New"] , inplace=True)

    df_data.loc[df_data["New"] == np.inf, "New"] = df_data["followers_count"]

    print(df_data)

    print(df_data["New"].mean())
