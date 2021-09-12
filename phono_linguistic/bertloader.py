import torch
import pickle
import os, random
from params import params
import numpy as np

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

MAX_LEN = 0

basepath = "/".join(os.path.realpath(__file__).split('/')[:-1])
DATA_PATH = os.path.join(basepath, "data")


class CustomDataset:
    def __init__(self):
        self.label2id = {'anger': 0, 'disgust': 1, 'fear': 2,
                        'joy': 3, 'neutral': 4, 'sadness': 5,
                        'surprise': 6
                    }

        self.id2label = {v: k for k,v in self.label2id.items()}
        print(self.label2id, "||", self.id2label)

        train_set, test_set = self.load_dataset(DATA_PATH)

        if params.dummy_run == True:
            self.train_dataset, self.criterion_weights = self.batched_dataset(train_set, dummy=True)
            self.eval_dataset = self.train_dataset
        else:
            print("Train_dataset:", end= " ")
            self.train_dataset, self.criterion_weights = self.batched_dataset(train_set)
            print("Test_dataset:", end= " ")
            self.eval_dataset, _ = self.batched_dataset(test_set)

    def load_dataset(self, path):
        # Load the dataset
        train_dataset = pickle.load(open(DATA_PATH + "/train.pkl", "rb"))
        test_dataset = pickle.load(open(DATA_PATH + "/test.pkl", "rb"))

        return train_dataset, test_dataset

    def batched_dataset(self, unbatched, dummy=False): # For batching full or a part of dataset.
        dataset = []
        criterion_weights = np.zeros(7) + 0.0000001 # 7 labels 

        idx = 0
        num_data = len(unbatched) if not dummy else 2

        while idx < num_data:
            batch_text = []
            batch_audio = []
            batch_text_mask = []
            batch_ids = []
            labels = []
            
            for data_idx in range(idx, min(idx+params.batch_size, num_data)):
                single_data = unbatched[data_idx]

                this_emo_ids = self.label2id[single_data["emotion"]]
                criterion_weights[this_emo_ids] += 1
                labels.append(this_emo_ids)

                batch_audio.append(single_data['input_values'][0][:(16000)*15])
                batch_text.append(single_data['input_values'][1])

                batch_text_mask.append(single_data['attention_mask'][1])

                batch_ids.append(single_data['filename'])

            texts = torch.LongTensor(batch_text)
            labels = torch.LongTensor(labels)
            text_masks = torch.LongTensor(batch_text_mask)

            max_audio_len = max(len(x) for x in batch_audio)
            audio = torch.FloatTensor([x + [0.0] * (max_audio_len - len(x))
                                        for x in batch_audio])
            audio_mask = torch.LongTensor([[1] * len(x) + [0] * (max_audio_len - len(x))
                                        for x in batch_audio])

            global MAX_LEN
            MAX_LEN = max(MAX_LEN, audio.shape[1])

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            t = texts.size(1)
            a = audio.size(1)

            assert texts.size() == torch.Size([b, t]) # Maxlen = 75
            assert text_masks.size() == torch.Size([b, t])
            assert labels.size() == torch.Size([b])
            assert audio.size() == torch.Size([b, a])
            assert audio_mask.size() == torch.Size([b, a])

            dataset.append((texts, audio, labels, text_masks, audio_mask, batch_ids))
            idx += params.batch_size

            if len(dataset) % 10 == 0:
                print(len(dataset), ", ")
        print("num_batches=", len(dataset), " | num_data=", num_data)
        criterion_weights = np.sum(criterion_weights)/criterion_weights
        print(MAX_LEN)
        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = CustomDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    import pickle
    pickle.dump(dataset, open('dataset_cache.pkl', 'wb'))
    
    print(len(dataset.train_dataset))#[0])
    print(dataset.train_dataset[-1])
    #print(len(dataset.hard_dataset))
    import os
    os.system("nvidia-smi")
    print(MAX_LEN)
