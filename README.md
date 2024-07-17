# QWER

```bash
├── classifier
│   ├── fine-tuning
│   ├── ft_dataset
│   │   ├── generation
│   │   └── labeling
│   │       └── auto_evaluation
│   │           ├── prompts
│   │           └── result
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
│   └── raw_data
│       ├── 2wikimultihopqa
│       ├── hotpotqa
│       ├── musique
│       ├── nq
│       ├── squad
│       ├── trivia
│       └── wiki
├── rewriter
└── scripts
```

## Installation
```bash
sh scripts/env_setup.sh
```

## Usage

### Classifier
```bash
sh scripts/CRAG_evaluate_for_labeling.sh
sh scripts/label_ft_dataset.sh
sh scripts/fine_tune_classifier.sh
```

### Rewriter

### Experiments

```bash
sh scripts/experimental_setup.sh
sh scripts/predict.sh
sh scripts/evaluate_RAG_predictions.sh
```