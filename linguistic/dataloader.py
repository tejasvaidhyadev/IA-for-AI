import torch
import os, json, random, sys
from params import params
import numpy as np
from transformers import AutoTokenizer

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

MAX_LEN = 0

LABEL_PATH = "../dataset"

class Dataset:
    def __init__(self):
        self.label2id = {'anger': 0,
                          'disgust': 1,
                          'fear': 2,
                          'joy': 3,
                          'neutral': 4,
                          'sadness': 5,
                          'surprise': 6
                         }

        self.id2label = {v: k for k,v in self.label2id.items()}
        print(self.label2id, "||", self.id2label)

        train_set, test_set = self.load_dataset(LABEL_PATH)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
        
        print("Loaded Bert Tokenizer")

        if params.dummy_run == True:
            self.train_dataset, self.criterion_weights = self.batched_dataset([train_set[0]] * 2)
            self.eval_dataset, _ = self.batched_dataset([train_set[0]] * 2)
        elif params.test_mode:
            print("Train_dataset:", end= " ")
            self.train_dataset, self.criterion_weights = self.batched_dataset(train_set)
            print("Eval_dataset:", end= " ")
            self.eval_dataset, _ = self.batched_dataset(test_set)
        else:
            split = (len(train_set) * 4) //5
            print(split)
            print("Train_dataset:", end= " ")
            self.train_dataset, self.criterion_weights = self.batched_dataset(train_set[:split])
            print("Eval_dataset:", end= " ")
            self.eval_dataset, _ = self.batched_dataset(train_set[split:])

        # self.criterion_weights = torch.tensor(self.criterion_weights.tolist()).to(params.device)
        # print("Training loss weighing = ", self.criterion_weights)

    def load_dataset(self, path):
        # Load the dataset
        train_text = json.load(open('train.json', "r"))
        test_text = json.load(open('test.json', 'r'))

        train_labels = [{'filename': x.strip().split(',')[0],
                        'filepath': "../datasets/TrainAudioFiles/" + x.strip().split(',')[0],
                        'label': x.strip().split(',')[1], 
                       	'text':	train_text[x.strip().split(',')[0]]
       	       	       }
                       for x in open(path + "/train.csv").readlines()[1:]
                     ]

        test_labels = [{'filename': x.strip(),
                        'filepath': "../datasets/TestAudioFiles/" + x.strip(),
                        'label': random.choice(list(self.label2id.keys())),
                        'text': test_text[x.strip()]
                       }
                       for x in open(path + "/test.csv").readlines()[1:] 
       	       	     ]

        assert len(train_labels) == len(train_text)
        assert len(test_labels) == len(test_text)

        return train_labels, test_labels

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []
        criterion_weights = np.zeros(7) + 0.0000001 # 7 labels 

        idx = 0
        num_data = len(unbatched)

        while idx < num_data:
            batch_text = []
            labels = []
            batch_ids = []
            
            for single_data in unbatched[idx:min(idx+params.batch_size, num_data)]:
                this_label_ids = self.label2id[single_data["label"]]
                criterion_weights[this_label_ids] += 1
                labels.append(this_label_ids)

                this_text = single_data['text']
                batch_text.append(this_text)

                batch_ids.append(single_data['filename'])

            tokenized_batch = self.bert_tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True,
                                                                return_tensors="pt", return_token_type_ids=True)

            texts = tokenized_batch['input_ids'].to(params.device)
            labels = torch.LongTensor(labels).to(params.device)
            pad_masks = tokenized_batch['attention_mask'].squeeze(1).to(params.device)

            global MAX_LEN
            MAX_LEN = max(MAX_LEN, texts.shape[1])

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            l = texts.size(1)
            assert texts.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP
            assert labels.size() == torch.Size([b])
            assert pad_masks.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP

            dataset.append((texts, labels, pad_masks, batch_ids))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        criterion_weights = np.sum(criterion_weights)/criterion_weights
        print(MAX_LEN)
        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = Dataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(len(dataset.train_dataset))#[0])
    print(dataset.train_dataset[-1])
    print(dataset.eval_dataset[-1])

    import os
    os.system("nvidia-smi")
    print(MAX_LEN)
