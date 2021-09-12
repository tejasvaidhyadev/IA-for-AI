import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=4214)
parser.add_argument("--test_mode", type=str, default="False")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--n_epochs", type=int, default=5)

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="Run the model on one sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--bert_type", type=str, required=True)
params = parser.parse_args()

params.test_mode = params.test_mode.lower() == "true"

