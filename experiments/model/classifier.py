from typing import Tuple

import torch
from transformers import pipeline

class Classifier:
    def __init__(self, model_name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = pipeline("text-classification", model=model_name, device=device)

    def classify(
        self,
        query: str
    ) -> Tuple[str, str]:
        output = self.pipeline(query)[0]
        label = output['label']
        score = output['score']
        return label, score
