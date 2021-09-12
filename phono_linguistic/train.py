
# Load Packages and setup wandb
from params import params
import wandb
if params.wandb:
    wandb.init(project="tejas", name=params.run)
    wandb.config.update(params)

from bertloader import CustomDataset
import json, os, random

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoConfig

from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for batch in dataset:
        
        (texts, audio, labels, text_masks, audio_mask, batch_ids) = batch
        preds = model(texts, text_masks, audio, audio_mask)
        loss = criterion(preds, labels.to(params.device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        if num_batch % 100 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion, target_names):
    raise NotImplementedError
    model.eval()
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset:
            (texts, stances, att_masks, token_type) = batch
            preds = model(texts, att_masks, token_type)

            loss = criterion(preds, stances)

            predicts.extend(torch.max(preds, axis=1)[1].tolist())
            gnd_truths.extend(stances.tolist())
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
import pickle
dataset_object = pickle.load(open("dataset_cache.pkl", 'rb'))
train_dataset = dataset_object.train_dataset
eval_dataset = dataset_object.eval_dataset

if params.dummy_run:
    train_dataset = [train_dataset[0]]
    eval_dataset = train_dataset
    target_names = []
else:
    eval_dataset = dataset_object.eval_dataset
    target_names = [dataset_object.id2label[id_] for id_ in range(0, 7)]


print("Dataset created")
os.system("nvidia-smi")


############# Create model #############

config = AutoConfig.from_pretrained(
            params.model_name_or_path,
            num_labels=7,
            label2id=dataset_object.label2id,
            id2label=dataset_object.id2label,
            finetuning_task="wav2vec2_clf",
            cache_dir=None,
            revision=None,
            use_auth_token=None,
        )

setattr(config, 'pooling_mode', "mean")

from models import BertHubertForSpeechClassification
model = BertHubertForSpeechClassification(
            params.bert_type,
            params.model_name_or_path,
            from_tf=False,
            config=config,
            revision=None,
            use_auth_token=False
        )
print("Model created")

print(sum(p.numel() for p in model.parameters()))
model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load("saved_models/29_20_29_53/model.pt"))
print("loading_previous model")
# hard coding for now 
# if params.init_model:
#     model.load_state_dict(torch.load(params.init_model))
#     print(params.init_model)
# print("sucessfully loaded state dict")


if params.wandb:
    wandb.watch(model)

########## Optimizer & Loss ###########

#criterion = torch.nn.CrossEntropyLoss(weight=dataset_object.criterion_weights, reduction='sum')
criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(params.device)
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

valid_loss = 0.0
# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    # if not params.dummy_run:
    #     print("EVALUATING:")
    #     valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)
    # else:
    #     valid_loss = 0.0

    if not params.dummy_run and params.wandb:
        wandb_dict = {}
        # for labl in target_names:
        #     for metric, val in classify_report[labl].items():
        #         if metric != "support":
        #             wandb_dict[labl + "_" + metric] = val

        # wandb_dict["F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
        # wandb_dict["F1-Avg"] = classify_report["macro avg"]["f1-score"]

        # wandb_dict["Accuracy"] = classify_report["accuracy"]

        wandb_dict["Train_loss"] = train_loss
        # wandb_dict["Valid_loss"] = valid_loss

        wandb.log(wandb_dict)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f}' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)


def predict_and_save(model, test_dataset):
    from datetime import datetime
    import os
    try:
        os.mkdir('saved_models')
    except:
        pass

    now = datetime.now()
    dir_name = 'saved_models/' + now.strftime("%d_%H_%M_%S/")
    os.mkdir(dir_name)

    model.eval()
    torch.save(model.state_dict(), dir_name + 'model.pt')

    all_logits = {}
    all_predicts = {}

    for batch in test_dataset:
        
        (texts, audio, _, text_masks, audio_mask, batch_ids) = batch
        batch_preds = model(texts, text_masks, audio, audio_mask)
        for i, p in zip(batch_ids, batch_preds.tolist()):
            all_logits[i] = p
            all_predicts[i] = dataset_object.id2label[int(np.argmax(p))]

    json.dump(all_logits, open(dir_name + "logits.json", 'w+'))
    json.dump(all_predicts, open(dir_name + 'predicts.json', 'w+'))
    open(dir_name + 'sample_submission.csv', 'w+').write(
            "filename,emotion\n" + \
            "\n".join([k.split('/')[-1] + ',' + v 
                    for k,v in all_predicts.items()]))
    json.dump(vars(params), open(dir_name + 'params.json', 'w+'))

if __name__ == "__main__":
    predict_and_save(model, eval_dataset)
