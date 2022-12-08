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
