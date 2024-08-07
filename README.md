# QWER

```
├── classifier
│   ├── fine-tuning
│   ├── ft_dataset
│   │   ├── evaluation
│   │   │   ├── metric
│   │   │   └── result
│   │   ├── labeling
│   │   ├── predictions
│   │   └── source
│   └── model
├── experiments
│   ├── metrics
│   ├── official_evaluation
│   │   ├── 2wikimultihopqa
│   │   ├── hotpotqa
│   │   └── musique
│   ├── predictions
│   │   ├── 2wikimultihopqa
│   │   ├── hotpotqa
│   │   ├── musique
│   │   ├── nq
│   │   ├── squad
│   │   └── trivia
│   ├── processed_data
│   │   ├── 2wikimultihopqa
│   │   ├── hotpotqa
│   │   ├── musique
│   │   ├── nq
│   │   ├── squad
│   │   └── trivia
│   ├── processing
│   ├── raw_data
│   │   ├── 2wikimultihopqa
│   │   ├── hotpotqa
│   │   ├── musique
│   │   ├── nq
│   │   ├── squad
│   │   └── trivia
│   └── rewritten_data
└── scripts
```

## Installation
```bash
conda create -n qwer python=3.9
conda activate qwer
sh scripts/env_setup.sh
```

## Usage

### Classifier
```bash
sh scripts/query_wise_evaluate.sh
sh scripts/query_wise_label.sh
sh scripts/fine_tune_classifier.sh
```

### Experiments

```bash
sh scripts/experimental_setup.sh
sh scripts/rewrite_data.sh
```

```bash
sh scripts/predict_processed_data.sh
sh scripts/evaluate_RAG_predictions.sh
```

```bash
sh scripts/predict_rewritten_data.sh
sh scripts/evaluate_RAG_rewritten_predictions.sh
```