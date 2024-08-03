"""
Answer metric
"""
import collections
import re
import string
import os
from typing import Tuple, List

import evaluate
import ftfy
from metric.metric import Metric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class F1BertScoreMetric(Metric):
    def __init__(
        self, 
        bertscore_model_type: str = "roberta-large",
        n_threads: int = os.cpu_count()
    ) -> None:
        self._bertscore_model_type = bertscore_model_type
        self._n_threads = n_threads
        
        self._bertscore = evaluate.load("bertscore")


    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ) -> Tuple[float, float]:
        predicted_answer = ftfy.fix_text(predicted_answer)
        ground_truth_answers = [ftfy.fix_text(e) for e in ground_truth_answers]

        assert isinstance(predicted_answer, str)
        assert isinstance(ground_truth_answers, (Tuple, List))
        f1_scores = metric_max_over_ground_truths(compute_f1, predicted_answer, ground_truth_answers)
        bertscore = metric_max_over_ground_truths(self.compute_bertscore_f1, predicted_answer, ground_truth_answers)

        return f1_scores, bertscore


    def compute_bertscore_f1(
        self,
        prediction: str,
        ground_truth: str,
    ) -> float:
        assert isinstance(prediction, str) and isinstance(ground_truth, str)

        results = self._bertscore.compute(
            predictions=[prediction],
            references=[ground_truth],
            lang="en",
            model_type=self._bertscore_model_type,
            nthreads=self._n_threads,
        )
        return results['f1'][0]