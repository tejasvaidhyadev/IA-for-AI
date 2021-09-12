# IA-for-AI

## Dependencies:
- Pytorch >= 1.7
- Huggingface's Transformers >=4.5
- Huggingface's Datasets
- torchaudio
- soundfile
- sounddevice
- scikit-learn
- scipy
- numpy

Additional dependencies for running the app:
`streamlit, plotly`

If you are training the AST model, then also download the following dependencies:
`matplotlib, numba, timm, zipp, wget, llvmlite`

## Training model

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

