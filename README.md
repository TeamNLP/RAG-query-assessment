# QWER

```bash
├── classifier
│   ├── fine-tuning
│   └── ft_dataset
│       ├── generation
│       └── labeling
│           └── auto_evaluation
│               ├── prompts
│               └── result
├── rewriter
├── official_evaluation
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
```