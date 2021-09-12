import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Load data
    df_train = pd.read_csv('train.csv')
    train_path = "/home/dsanyal/.hack/dataset/TrainAudioFiles"
    # append the string "path" in the column filename in the df_train
    df_train['filename'] = df_train['filename'].map(lambda x: "/home/dsanyal/.hack/dataset/TrainAudioFiles/"+ str(x))

    test_path = "/home/dsanyal/.hack/dataset/TestAudioFiles"
    df_test = pd.read_csv('test.csv')
    df_test['filename'] = df_train['filename'].map(lambda x: "/home/dsanyal/.hack/dataset/TestAudioFiles/"+ str(x))

    # create a valid set of df_train into two part
    # 1. train_set: 80% of the df_train
    # 2. valid_set: 20% of the df_train
    # shuffle the df_train
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    train_set_size = int(len(df_train) * 0.8)
    train_set = df_train[:train_set_size]
    valid_set = df_train[train_set_size:]
    # save the valid set
    valid_set.to_csv('valid_set.csv', index=False)
    # save the train set
    train_set.to_csv('train_set.csv', index=False)
    # save the test set
    df_test.to_csv('test_set.csv', index=False)
