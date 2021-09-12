# IA-for-AI

![Approach](https://github.com/tejasvaidhyadev/IA-for-AI/blob/main/approach.png?raw=true)

## Dependencies:

We exported our conda environments for training the models and running the app.

- `train_env.yml`: Our environment for training. Create using `conda env create --name prompt --file=train_env.yml` & `conda activate prompt`
- `app_env.yml`: Our environment for app. Create using `conda env create --name prompt_app --file=app_env.yml`	& `conda activate prompt_app`

*Please note that our `app_env` was ran on a MacOS 11.2 Machine with Intel processor, whereas our training (`train_env`) was done on a linux machine with Nvidia GPUs. The same conda environment may not work on other machines. Instead you may download the packages individually.*

You may also download the following dependencies individually as an alternate means to create the environment:
- Pytorch
- Huggingface's Transformers
- Huggingface's Datasets
- torchaudio
- soundfile
- sounddevice
- scikit-learn
- scipy
- numpy

Additional dependencies for running the webapp: `streamlit, plotly`
- Create a fresh conda environment
- `pip install streamlit`
- `pip install soundfile`
- `pip install sounddevice`
- `pip install pydub`
- [Install PyTorch-1.8 and torchaudio](https://pytorch.org/get-started/previous-versions/)
- `pip install transformers==4.10`


If you are training the AST model, then also download the following dependencies:
`matplotlib, numba, timm, zipp, wget, llvmlite`

## Running the webapp
 
 ![WebApp](https://github.com/tejasvaidhyadev/IA-for-AI/blob/main/demo.jpeg?raw=true)
 
- Download the required dependencies or replicate & activate our conda environment, as detailed above.
- Our webapp is in the `app` folder: `cd app`.
- Download the [pre-trained model weights](https://drive.google.com/file/d/1KFSAYqRBzEkodBr7xirHkeG4bgGxJyz-/view?usp=sharing) and save it in the `webapp` folder with the name `cpu_model.pt`
- Run webapp: `streamlit run app.py`
- Note that it may take some time to run first time as models, tokenizers, feature-extractors and config are downloaded.
- The webapp will be hosted locally (port and address printed on command line).


## Training model

- Download the required dependencies or replicate our conda environment, as detailed above.
- Download the set of audio files in `TrainAudioFiles` & `TestAudioFiles` and place it inside the `dataset` folder.
- Get pseudo ASR ground truth labels by running `python3 run_asr.py` from inside the `linguistic` folder.
- If you want to train the linguistic (text-only) model, then you can run `python3 train.py` from inside `linguistic` folder with the following optional command line arguments .
  - Batch Size: `--batch_size=16`
  - Learning Rate: `--lr=1e-5`
  - Number of Epochs: `--n_epochs=10`
  - Do dummy run for debugging purposes: `--dummy_run`
  - Device to training the model on: `--device=cuda`
  - Randomizing seed: `--seed=1`
  - Test the model: `--test_model`
  - Bert Model name or path: `--bert_type=bert-base-uncased`
- For training AST: Refer to the instructions in `ast/README.md`, the dataset needs to downloaded and kept inside `ast/egs/hack/data/`.
- If you want to train a model only on audio features, then from inside `phono` folder, run `bash run.sh`. You may edit the `run.sh` to change the following arguments:
  - Pooling method to extract model features: `--pooling_mode` Options: ["mean",'max', 'sum']
  - Name or path of audio only model's pretrained file: `--model_name_or_path`
  - Type of model to use: `--model_mode` Example arguments: `hubert` or `wav2vec2`
  - Training Batch size per device: `--per_device_train_batch_size` [Type: Integer]
  - Eval Batch Size per device: `--per_device_eval_batch_size` [Type: Integer]
  - Learning rate: `--learning_rate`
  - Number of epochs: `--num_train_epochs`
  - Gradient Accumulation steps: `--gradient_accumulation_steps` (set 1 for no accumulation)
- Save, eval and logger steps: `--save_steps`, `eval_steps`, `logging_steps`
  - Maximum number of models to save: `--save_total_limit`
  - Required Arguments (do not change): `--freeze_feature_extractor`, `--input_column=filename`, `--target_column=emotion`, `output_dir="output_dir"` `delimiter="comma"`, `--evaluation_strategy="steps"`, `--fp16`, `--train_file="./../dataset/train_set.csv"`, `--validation_file="./../dataset/valid_set.csv"`, `--test_file="./../dataset/test_set.csv"`
  - Whether to do train, eval and predict on test: `--do_eval`, `--do_train`, `--do_predict`
- If you want to train phono_linguistic transformer model:
    - Extract phonetic features for `train` and `test` set from `phono_feat_extractor`: 
      - After obtaining ASR predictions from `linguistic`, put the `test.json` and `train.json` in the `phono_feat_extractor` folder.
      - Merge the above two files using `python3 merge_text.py` inside the `phono_feat_extractor` folder.
      - Then do `bash run.sh` from inside the same folder. You may change the same arguments as above mentioned for only on audio features. Additional argument of the Bert model: --bert_name='bert-base-uncased'.
    - After the above step, you will obtain the `train.pkl` and `test.pkl` files inside `phono_feat_extractor`. Put these files in `phono_linguistic/data` folder.
    - Run `python3 bertloader.py` from `phono-linguistic` folder to cache the dataloader for training.
    - And then train model using `python3 train.py`.
    - You may include the following arguments for `python3 bertloader.py` and `python3 train.py`. Make sure the same arguments are passed to the two commands
      - Seed: `--seed=1` (type=int)
      - Batch_Size: `--batch_size=16` (type=int)
      - Learning rate: `--lr=1e-5` (type=float)
      - Number epochs: `--n_epochs=5` (type=int)
      - Dummy run (for debugging purposes): `--dummy_run`
      - Device to train model on: `--device`
      - Whether to log on wanbd: `--wandb`
      - Bert model name or path: `--bert_type=bert-base-uncased`
      - Audio model name or path: `--model_name_or_path='facebook/hubert-large-ll60k'`

