import csv
import json
import os
import numpy as np

corrupt_files = [x.strip() for x in open("corrupt_files.txt").readlines()]

label_set = np.loadtxt('./data/esc_class_labels_indices.csv', delimiter=',', dtype='str')

label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

try:
    os.mkdir("./data/datafiles")
except:
    pass

train_data = np.loadtxt('./data/dataset/train.csv', delimiter=',', dtype='str', skiprows=1)
train_wav_list = []

for i in range(0, len(train_data)):
    if train_data[i][0] in corrupt_files:
        print("Skipping " + train_data[i][0])
        continue
    cur_label = label_map[train_data[i][1]]
    cur_path = './data/dataset/TrainAudioFiles/' + train_data[i][0]

    # hackrwj is just a dummy prefix
    cur_dict = {"wav": cur_path, "labels": 'hackrwj' + cur_label.zfill(2)}
    train_wav_list.append(cur_dict)

split = (len(train_wav_list) * 4) // 5

with open('./data/datafiles/hack_train_data.json', 'w+') as f:
    json.dump({'data': train_wav_list[:split]}, f, indent=1)

with open('./data/datafiles/hack_eval_data.json', 'w+') as f:
    json.dump({'data': train_wav_list[split:]}, f, indent=1)

print("Dataset size of train and test:", split, len(train_wav_list)-split)

print('Finished Hack Dataset Preparation')



exit()

test_data = np.loadtxt('./data/dataset/test.csv', delimiter=',', dtype='str', skiprows=1)
test_wav_list = []

for i in range(0, len(test_data)):
    cur_label = label_map[test_data[i][1]]
    cur_path = './data/dataset/TestAudioFiles/' + test_data[i][0]
    
    # hackrwj is just a dummy prefix
    cur_dict = {"wav": cur_path, "labels": 'hackrwj' + cur_label.zfill(2)}
    test_wav_list.append(cur_dict)

with open('./data/datafiles/hack_test_data.json', 'w+') as f:
    json.dump({'data': test_wav_list}, f, indent=1)

print("Dataset size of test:", len(test_wav_list))

print('Finished Hack Dataset Preparation')
