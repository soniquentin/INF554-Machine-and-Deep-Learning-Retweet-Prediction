# Retweet prediction

## Introduction

Retweet prediction challenge on Kaggle by Paul Hellegouarch, Yann Vastel and Quentin Lao

## Run commands

1) First, run the following command on the project directory to download the Python library we used `pip install -r requirements.txt`.

2) Make sure that file `train.csv` and `evaluation.csv` are present in the `data` folder.

3) Run the command `python main.py`. This will launch a Random Forest training, save the model and write the csv submission file.


## Modify the Random Forest's Hyperparameters

Open the main file `main.py`. At the bottom of the script, you'll find :
```
if __name__ == "__main__":
    ##==========================================##

    model_name = "super_manNN2"

    #### ========  RF  ========
    scaler = train_model(alg = "RF",
                        model_name = model_name ,
                        #loss = "absolute_error",  #GB
                        bootstrap = True,  #RF
                        max_depth = 20,
                        max_features = 1.0,
                        min_samples_leaf = 2,
                        min_samples_split = 2,
                        n_estimators = 500,
                        #objective = "reg:absoluteerror", #XGB
                        #eval_metric = "mae" #XGB
                        random_state = 41)
```

## Final Rank in the competition

We reach the Top 10 out of 58 teams.

| #             | Team                      | Score    |
| ------------- | ------------------------- | -------- |
| 1             |  CaFAy                    | 5.09983  |
| 2             |  Royals never give up     | 5.15101  |
| 3             |  Debug                    | 5.22905  |
| 4             |  GetSomeFries             | 5.26624  |
| 5             |  Ragnarok                 | 5.32145  |
| 6             |  BTS                      | 5.32673  |
| 7             |  Hexa                     | 5.38552  |
| 8             |  Team Lebron              | 5.42237  |
| 9             |  Les Kaggers              | 5.43725  |
| 10            |  Les hackers de l'extrême | 5.47227  |
