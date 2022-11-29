import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


train_file = "data/train.csv"

list_of_features = ["text",
                    "retweets_count",
                    "favorites_count",
                    "followers_count",
                    "statuses_count",
                    "friends_count",
                    "mentions",
                    "urls",
                    "verified",
                    "hashtags",
                    "timestamp",
                    "TweetID"]

X_train = pd.read_csv(train_file)

X_train.drop(columns = ["text", "urls", "mentions", "hashtags", "TweetID"], inplace = True)


#sns.distplot(X_train['retweets_count'])
sns.pairplot(X_train.head(100000))
plt.savefig('plot_result.png', dpi = 300)
