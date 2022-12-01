"""
    FUNCTIONS FOR FEATURES EXTRACTION
"""


def numberize_features(X, debug = True):
    """
        Transform all the string features (e.g list of links) into len

        UTILIZATION :
            X = numberize_features(X)
    """
    if debug :
        print("    --> numberize_features ")

    def str_to_len(x):
        if type(x) == str:
            return len(x.split("'")) -1
        else:
            return x

    return X.applymap(str_to_len)


def followers_over_friends(X, debug = True) :
    """
        Add the feature [FOLLOWERS COUNT]/[FRIENDS COUNT]
        Feature's name : fof
    """
    import numpy as np

    if debug :
        print("    --> followers_over_friends")


    X["fof"] = X["followers_count"]/X["friends_count"]

    #When friends_count = 0, fof = inf. We have to treat these cases

    # FIRST POSSIBILITY : replace inf by the mean
    #X.replace([np.inf], np.nan , inplace=True)
    #X.replace([np.nan], X["fof"].mean() , inplace=True)

    # SECOND POSSIBILITY : replace by the number of followers (as if friends_count = 1)
    X.loc[X["fof"] == np.inf, "fof"] = X["followers_count"]

    return X
