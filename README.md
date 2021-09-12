# IA-for-AI

## Dependencies for Training:
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
- streamlit
- plotly

## Training model

- Download the dataset and place it in `dataset`
- Get pseudo ASR ground truth labels by running `python3 run_asr.py` from inside the `linguistic` folder.
- If you want to train the linguistic (text-only) model, then you can run `python3 train.py` from inside `linguistic` folder with the following optional command line arguments .
  - Batch Size: `batch_size`
  - Learning Rate: `lr`
  - Number of Epochs: `n_epochs`
  - Do dummy run for debugging purposes: `dummy_run`
  - Device to training the model on: `device`
  - Randomizing seed: `seed`
  - Test the model: `test_model`
  - Bert Model name or path: `bert_type`

parser.add_argument("--bert_type", type=str, required=True)

For training Audio-only Hubert: run `bash run.sh` from it's folder.
For training AST: Refer to the instructions in ast/README.md
For training 


