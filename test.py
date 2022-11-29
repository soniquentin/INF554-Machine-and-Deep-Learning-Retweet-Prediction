from textblob import TextBlob
import sys
import os
from os.path import exists



if __name__ == "__main__":
    print(os.path.dirname(__file__))
    filename = "plot_results.png"

    file_exists = exists(os.path.dirname(__file__) + "/output/" + filename)

    print(file_exists)
