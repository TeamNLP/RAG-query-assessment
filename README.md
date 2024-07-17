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
│   ├── processed_data
│   │   ├── 2wikimultihopqa
│   │   ├── hotpotqa
│   │   ├── iirc
│   │   ├── musique
│   │   ├── nq
│   │   ├── squad
│   │   └── trivia
│   ├── processing
│   └── raw_data
│       ├── 2wikimultihopqa
│       ├── hotpotqa
│       ├── iirc
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
sh scripts/env_reproduction.sh
```

## Usage

### Classifier
```bash
sh scripts/evaluation_for_labeling.sh
sh scripts/ft_dataset_labeling.sh
sh scripts/fine_tuning_classifier.sh
```

### Rewriter

### Experiments

```bash
sh scripts/experimental_setup.sh
sh scripts/predict.sh
sh scripts/evaluate_prediction.sh
```