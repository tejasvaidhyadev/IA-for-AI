from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        #self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
        ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        return outputs,hidden_states

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertHubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, bert_config, speech_config, num_labels):
        super().__init__()
        self.dense = nn.Linear(bert_config.hidden_size + speech_config.hidden_size,
                                bert_config.hidden_size + speech_config.hidden_size
                            )
        self.dropout = nn.Dropout(speech_config.final_dropout)
        self.out_proj = nn.Linear(bert_config.hidden_size + speech_config.hidden_size,
                                    num_labels
                                )

    def forward(self, bert_features, speech_features, **kwargs):
        x = torch.cat([bert_features, speech_features], 1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BertHubertForSpeechClassification(nn.Module):
    def __init__(self, bert_name, audio_model_name, from_tf=None, config=None, revision=None, use_auth_token=None):
        super(BertHubertForSpeechClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.audio_model = Wav2Vec2ForSpeechClassification.from_pretrained(#ubertForSpeechClassification.from_pretrained(
            audio_model_name,
            from_tf=from_tf,
            config=config,
            revision=revision,
            use_auth_token=use_auth_token
        )
        # to_remove = ['classifier.dense.weight', 'classifier.dense.bias',
        #              'classifier.out_proj.weight', 'classifier.out_proj.bias'
        #             ]
        # loaded_state_dict = torch.load('saved_models/checkpoint-23000/pytorch_model.bin',
        #                     map_location=torch.device('cpu'))
        # for x in to_remove:
        #     _ = loaded_state_dict.pop(x)

        # self.audio_model.load_state_dict(loaded_state_dict)

        self.classifier = BertHubertClassificationHead(self.bert.config,
                                    self.audio_model.config,
                                    self.audio_model.config.num_labels
                                )
        self.num_labels = self.audio_model.num_labels

    def freeze_feature_extractor(self):
        self.audio_model.freeze_feature_extractor()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
        ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            bert_input_values,
            bert_attention_mask,
            input_values,
            attention_mask
        ):
        assert input_values != None and bert_input_values != None
        # assert bert_attention_mask != None
        _, audio_features = self.audio_model(input_values,
                            attention_mask=attention_mask,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=False,
                        )
        bert_outputs = self.bert(
            bert_input_values,
            attention_mask=bert_attention_mask,
        )

        logits = self.classifier(audio_features, bert_outputs.pooler_output)
        return logits


# label2id = {'anger': 0, 'disgust': 1, 'fear': 2,
#                         'joy': 3, 'neutral': 4, 'sadness': 5,
#                         'surprise': 6
#                     }
# id2label = {v: k for v,k in label2id.items()}
# config = AutoConfig.from_pretrained(
#             'facebook/hubert-large-ll60k',
#             num_labels=7,
#             label2id=label2id,
#             id2label=id2label,
#             finetuning_task="wav2vec2_clf",
#             cache_dir=None,
#             revision=None,
#             use_auth_token=None,
#         )

# setattr(config, 'pooling_mode', "mean")

# model = BertHubertForSpeechClassification(
#             "bert-base-uncased",
#             'facebook/hubert-large-ll60k',
#             from_tf=False,
#             config=config,
#             revision=None,
#             use_auth_token=False
#         )
# print("Model created")
