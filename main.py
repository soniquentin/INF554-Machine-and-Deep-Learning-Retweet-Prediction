from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tools import *



train_file = "data/train.csv"
test_file = "data/evaluation.csv"

X_test =  pd.read_csv(test_file)

X_train = pd.read_csv(train_file)
y_train = X_train["retweets_count"]
##X_train.drop(['retweets_count'], axis = 1, inplace = True )

"""
print("NUMBERIZE")
X_train = numberize_features(X_train)
"""


print("Sentiment preccessing")
X_train = sentiment_processing(X_train, "text")
print("Plotting")
plot(X_train)


"""
X_test = numberize_features(X_test)

print("TRAIN")
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

print("PREDICT")
predictions = rf.predict(X_test)

X_test["predictions"] = predictions

with open('data/rf3.txt', 'w') as f:
    f.write('TweetID,retweets_count\n')
    for index, row in X_test.iterrows():
        f.write('{},{}\n'.format(row["TweetID"], int(row["predictions"]) ))
"""

"""
print("TRAIN")

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train2, y_train2)
evaluation(rf, X_test2, y_test2)
"""
