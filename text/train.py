# Load Packages and setup wandb
from params import params

from dataloader import Dataset
import json, os, random

import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW, AutoModel

from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for batch in dataset:
        (texts, labels, att_masks, _) = batch
        # print(texts, att_masks)
        preds = model(texts, att_masks)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if num_batch % 10 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion, target_names):
    model.eval()
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset:
            (texts, labels, att_masks,_) = batch
            preds = model(texts, att_masks)

            loss = criterion(preds, labels)

            predicts.extend(torch.max(preds, axis=1)[1].tolist())
            gnd_truths.extend(labels.tolist())
            valid_losses.append(loss.item())

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    if params.dummy_run:
        classify_report = {"hi": {"fake": 1.2}}
    else:
        classify_report = classification_report(gnd_truths, predicts, target_names=target_names, output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    for labl in target_names:
        print(labl,"F1-score:", classify_report[labl]["f1-score"])
    print("Accu:", classify_report["accuracy"])
    print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
    print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report


########## Load dataset #############
dataset_object = Dataset()
train_dataset = dataset_object.train_dataset
eval_dataset = dataset_object.eval_dataset

if params.dummy_run:
    eval_dataset = train_dataset
    target_names = []
else:
    eval_dataset = dataset_object.eval_dataset
    target_names = [dataset_object.id2label[id_] for id_ in range(0, 7)]


print("Dataset created")
os.system("nvidia-smi")


########## Create model #############

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=7):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(params.bert_type)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, text, att_mask):
        output = self.bert(text, attention_mask=att_mask)
        return self.classifier(self.drop(output['pooler_output']))

model = BERTClassifier(7)
print("Model created")
os.system("nvidia-smi")

print(sum(p.numel() for p in model.parameters()))
model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")

########## Optimizer & Loss ###########

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    if not params.dummy_run:
        print("EVALUATING:")
        valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)
    else:
        valid_loss = 0.0

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f}' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

import datetime

if params.test_mode:
    save_folder = "expt/" + datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
    
    os.mkdir(save_folder)

    # Store params
    json.dump(vars(params), open(os.path.join(save_folder, "params.json"), 'w+'))

    # Save model
    torch.save(model.state_dict(), os.path.join(save_folder, "model.pt"))

    predictions = {}
    model.eval()
    with torch.no_grad():
        for batch in eval_dataset:
            (texts, labels, att_masks, ids) = batch
            preds = model(texts, att_masks)

            predicts = torch.max(preds, axis=1)[1].tolist()
            for id_, pred_ in zip(ids, predicts):
                predictions[id_] = pred_

    json.dump(predictions, open(os.path.join(save_folder, "predictions.json"), 'w+'), indent=1)

