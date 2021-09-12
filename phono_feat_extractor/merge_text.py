import json
train_text = "train.json"
test_text = "test.json"
jo = lambda x: json.load(open(x))

dataset = jo(train_text)
dataset.update(jo(test_text))
json.dump(datasets, open('text.json','w+'))
