import streamlit as st

import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import tempfile
from pydub import AudioSegment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AutoTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC
from my_models import BertHubertForSpeechClassification

from libs.dummy import outputs as dummy_outputs
from libs.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
from libs.utils import (
    set_session_state,
    get_session_state,
    local_css,
    remote_css,
    plot_result
)

import meta


class SpeechToText:
    def __init__(
            self,
            ctc_model_name="facebook/wav2vec2-base-960h",
            cf_model_name="facebook/wav2vec2-base-960h",
            ctc_model_type="wav2vec2",
            cf_model_type="wav2vec2",
            device=0,
            channels=1,
            subtype="PCM_24",
        ):
        self.label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.id2label = {v: k for k,v in self.label2id.items()}

        self.debug = False
        self.dummy_outputs = dummy_outputs

        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctc_model_name = ctc_model_name
        self.cf_model_name = cf_model_name
        self.ctc_model_type = ctc_model_type
        self.cf_model_type = cf_model_type

        self.device = device
        self.device_info = sd.query_devices(device, 'input')
        self.samplerate = int(self.device_info['default_samplerate'])
        self.channels = channels
        self.subtype = subtype

        self.cf_feature_extractor = None
        self.cf_config = None
        self.cf_samplerate = None
        self.cf_model = None

        self.ctc_processor = None
        self.ctc_samplerate = None
        self.ctc_model = None

    def recording(self, duration_in_seconds=10):
        recording = sd.rec(
            frames=int((duration_in_seconds + 0.5) * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            blocking=True,
        )
        sd.wait()
        return recording

    def load_cf(self):
        config = AutoConfig.from_pretrained(
            'facebook/hubert-large-ll60k',
            num_labels=7,
            label2id=self.label2id,
            id2label=self.id2label,
            finetuning_task="wav2vec2_clf",
            cache_dir=None,
            revision=None,
            use_auth_token=None,
        )

        setattr(config, 'pooling_mode', "mean")

        self.model = BertHubertForSpeechClassification(
                "bert-base-uncased",
                'facebook/hubert-large-ll60k',
                from_tf=False,
                config=config,
                revision=None,
                use_auth_token=False
            )

        print("Model created")
        self.model.load_state_dict(torch.load('cpu_model.pt'))
        self.model.eval()
        print("Model loaded")
        self.model.freeze_feature_extractor()
        
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print("Tokenizer loaded")
        print(sum(p.numel() for p in self.model.parameters()))
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
        # config = AutoConfig.from_pretrained(self.cf_model_name)

        # if self.cf_model_type == "wav2vec":
        #     model = Wav2Vec2ForSpeechClassification.from_pretrained(self.cf_model_name).to(self.torch_device)
        # elif self.cf_model_type == "hubert":
        #     model = HubertForSpeechClassification.from_pretrained(self.cf_model_name).to(self.torch_device)
        # else:
        #     model = Wav2Vec2ForSpeechClassification.from_pretrained(self.cf_model_name).to(self.torch_device)

        self.cf_feature_extractor = feature_extractor
        # self.cf_config = config
        self.cf_samplerate = feature_extractor.sampling_rate
        self.cf_model = self.model

    def load_ctc(self):
        processor = Wav2Vec2Processor.from_pretrained(self.ctc_model_name)

        if self.ctc_model_type == "wav2vec":
            model = Wav2Vec2ForCTC.from_pretrained(self.ctc_model_name).to(self.torch_device)
        elif self.ctc_model_type == "hubert":
            model = HubertForCTC.from_pretrained(self.ctc_model_name).to(self.torch_device)
        else:
            model = Wav2Vec2ForCTC.from_pretrained(self.ctc_model_name).to(self.torch_device)

        self.ctc_processor = processor
        self.ctc_samplerate = processor.feature_extractor.sampling_rate
        self.ctc_model = model

    def load(self):
        if not self.debug:
            self.load_cf()
            self.load_ctc()

    def _speech_file_to_array_fn(self, path, samplerate):
        speech_array, sr = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def predict_cf(self, path, text):
        speech_list = [self._speech_file_to_array_fn(path, self.samplerate)]
        text_list = [self.bert_tokenizer(text.lower())['input_ids'][:75]]
        result = self.cf_feature_extractor(speech_list, sampling_rate=16000)#self.samplerate)

        # print(result.keys(), '\n\n\n\n')
        # result['input_values'] = [[x, y] for x, y in zip(result['input_values'], text_list)]
        # result['attention_mask'] = [[x, y] for x, y in zip(result['attention_mask'],)]
        # result['text_list'] = text_list
        # result['text_pad_list'] = text_pad_list
        print(result.keys(), len(result['input_values']), '\n\n\n\n')

        text_ip = torch.LongTensor(text_list)
        audio_ip = torch.FloatTensor(result['input_values'])
        print(audio_ip.shape)
        print(text_ip.shape)
        with torch.no_grad():
            batch_preds = self.cf_model(text_ip, None, audio_ip, None)

        print(batch_preds)
        scores = F.softmax(batch_preds, dim=1).detach().cpu().numpy()[0]
        print(scores)
        outputs = [{"label": self.id2label[i],
                    "score": float(score)
                    } for i, score in enumerate(scores)
                ]

        return outputs
        speech = self._speech_file_to_array_fn(path, self.cf_samplerate)
        inputs = self.cf_feature_extractor(speech, sampling_rate=self.cf_samplerate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.torch_device) for key in inputs}
        return {'label': 'anger', 'score': [1]}

        with torch.no_grad():
            logits = self.cf_model(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [
            {
                "label": self.cf_model.config.id2label[i],
                "score": score
            } for i, score in enumerate(scores)
        ]
        return outputs

    def predict_ctc(self, path):
        speech = self._speech_file_to_array_fn(path, self.ctc_samplerate)
        inputs = self.ctc_processor(speech, sampling_rate=self.ctc_samplerate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.torch_device) for key in inputs}

        with torch.no_grad():
            logits = self.ctc_model(**inputs).logits

        return self.ctc_processor.batch_decode(torch.argmax(logits, dim=-1))

    def predict(self, path):
        if self.debug:
            return self.dummy_outputs

        ctc = self.predict_ctc(path)
        cf = self.predict_cf(path, ctc[0] if len(ctc) > 0 else ctc)

        return {
            "ctc": ctc[0] if len(ctc) > 0 else ctc,
            "cf": cf
        }


@st.cache(allow_output_mutation=True)
def load_tts():
    tts = SpeechToText(
        ctc_model_name="facebook/wav2vec2-large-960h-lv60",
        # cf_model_name="m3hrdadfi/hubert-base-persian-speech-emotion-recognition",
        cf_model_name="m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition",
        ctc_model_type="wav2vec",
        # cf_model_type="hubert",
        cf_model_type="wav2vec",
    )
    tts.load()
    return tts

def main():
    st.set_page_config(
        page_title="Prompt Engineer",
        page_icon="<3",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    remote_css("https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font/dist/font-face.css")
    set_session_state("_is_recording", False)
    local_css("assets/style.css")
    # st.write(f"DEVICES: {sd.query_devices()}")

    tts = load_tts()

    col1, col2 = st.columns([5, 7])
    with col2:
        st.markdown('<div class="mt"></div>', unsafe_allow_html=True)
        st.markdown('<br><br><br>', unsafe_allow_html=True)
        audio_player = st.empty()
        speech_text = st.empty()

    with col1:
        st.markdown(meta.INFO, unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload an Audio File",type=['wav'])
        duration = st.slider('Choose your recording duration (seconds)', 5, 20, 5)
        recorder_btn = st.button("Recording")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    info = st.empty()

    if recorder_btn:
        if not get_session_state("_is_recording"):
            set_session_state("_is_recording", True)

            info.info(f"{duration} of Recording in seconds ...")
            np_audio = tts.recording(duration_in_seconds=duration)
            if len(np_audio) > 0:
                filename = tempfile.mktemp(prefix='tmp_sf_', suffix='.wav', dir='')
                with sf.SoundFile(
                        filename,
                        mode='x',
                        samplerate=tts.samplerate,
                        channels=tts.channels,
                        subtype=tts.subtype
                ) as tmp_audio:
                    tmp_audio.write(np_audio)

                audio_player.audio(filename)
                speech_text.info(f"Converting speech to text ...")
                result = tts.predict(filename)
                speech_text.markdown(
                    f'<p class="ctc-box ltr"><strong>Text: </strong>{result["ctc"]}</p>',
                    unsafe_allow_html=True
                )

                info.info(f"Recognizing emotion ...")
                plot_result(result["cf"])

                if os.path.exists(filename):
                    os.remove(filename)

                info.empty()
                set_session_state("_is_recording", False)

    if audio_file is not None:
        # file_details = {"FileName":audio_file.name,"FileType":audio_file.type}
        # st.write(file_details)
        with open(audio_file.name,"wb") as f:
            f.write(audio_file.getbuffer())
        st.success("Saved File")

        # if audio_file.name.endswith('.wav'):
        sound = AudioSegment.from_wav(audio_file.name)
        # else:
            # sound = AudioSegment.from_mp3(audio_file.name)
        sound = sound.set_channels(1)
        sound.export(audio_file.name, format="wav")

        # print(type(audio_file))
        audio_player.audio(audio_file.name)
        speech_text.info(f"Converting speech to text ...")
        result = tts.predict(audio_file.name)
        speech_text.markdown(
            f'<p class="ctc-box ltr"><strong>Text: </strong>{result["ctc"]}</p>',
            unsafe_allow_html=True
        )

        info.info(f"Recognizing emotion ...")
        plot_result(result["cf"])

        if os.path.exists(audio_file.name):
            os.remove(audio_file.name)

            info.empty()



if __name__ == '__main__':
    main()
