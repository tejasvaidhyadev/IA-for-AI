#import soundfile as sf
import torch
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
import os
import librosa
import json

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-large-960h-lv60").cuda()
model.eval()

def transcribe(basepath, savepath):
    files = os.listdir(basepath)
    text = {}

    for i, file in enumerate(files):
        if i % 10 == 0:
            print("\n\n========", i, file, "=======\n\n")

        # load audio
        audio_input, _ = librosa.load(basepath + "/" + file, sr = 16000)

        # transcribe
        input_values = tokenizer(audio_input, return_tensors="pt").input_values
        logits = model(input_values.cuda()).logits
        predicted_ids = torch.argmax(logits, dim=-1).cpu()
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        text[file] = transcription

    json.dump(text, open(savepath, "w+"), indent=1)

with torch.no_grad():
    transcribe("../dataset/TestAudioFiles", "test.json")

with torch.no_grad():
    transcribe("../dataset/TrainAudioFiles", "train.json")
