from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def numberize_features(X):
    for index, row in X.iterrows():

        for col in X.columns :
            if "str" in str(type(row[col])) :
                X.at[index, col] = row[col].count("'")

    return X



def evaluation(rf, X_test, y_test):
    pred = rf.predict(X_test)
    for i in range(len(pred)):
        pred[i] = round(pred[i])
    print("Mean square error : {}".format( mean_squared_error(y_test, pred)) )



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
