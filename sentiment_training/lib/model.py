import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoConfig


class Roberta(nn.Module):
    def __init__(self, model_type, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_type)
        self.config = AutoConfig.from_pretrained(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        # Tiny MLP for classification, 7 classes is hard coded
        self.linear1 = nn.Linear(self.config.hidden_size, 100)
        self.linear2 = nn.Linear(100, 7)
        self.classifier_dropout = nn.Dropout(0.1)

    def forward(self, sentences: List[str]) -> torch.FloatTensor:
        """
        Compute sentiment predictions given a list of sentences.
        Tokenisation will happen in place.
        Args:
            - sentences (list of str)
        Returs:
            - prediction (FloatTensor, shape is bsz x 7)
        """
        encoded_input = self.tokenizer(
            list(sentences), return_tensors='pt', padding=True)
        for k in encoded_input:
            if torch.cuda.is_available():
                encoded_input[k] = encoded_input[k].cuda()

        # Compute hidden reps and classify based on CLS token
        hidden = self.model(**encoded_input)[0]
        hidden = self.classifier_dropout(self.linear1(hidden[:, 0, :]))
        prediction = torch.log_softmax(self.linear2(
            torch.relu(hidden)).squeeze(-1), dim=-1)
        return prediction
