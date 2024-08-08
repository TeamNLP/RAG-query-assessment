from typing import Tuple

import torch
from transformers import pipeline

class Classifier:
    def __init__(self, model_name, device=None):
        if device=None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.pipeline = pipeline("text-classification", model=model_name, device=self.device)

    def classify(
        self,
        query: str
    ) -> Tuple[str, str]:
        output = self.pipeline(query)[0]
        label = output['label']
        score = output['score']
        return label, score
