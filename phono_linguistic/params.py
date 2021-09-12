import argparse
from typing_extensions import ParamSpec

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=4214)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--n_epochs", type=int, default=2)

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
parser.add_argument("--bert_type", type=str, default="bert-base-uncased")
parser.add_argument("--model_name_or_path", type=str, default='facebook/hubert-large-ll60k')

params = parser.parse_args()

# params.bert_type = "bert-base-uncased"
