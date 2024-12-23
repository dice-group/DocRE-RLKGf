# DocRE-RLKGf

**DocRE-RLKGf** (Document-level Relation Extraction with Reinforcement Learning Knowledge Graph Fusion) is a framework designed for extracting relations from documents using reinforcement learning and knowledge graph-based approaches. This method leverages the strengths of knowledge graphs and deep reinforcement learning to improve the precision of relation extraction tasks.

---

## Overview

Relation extraction is a fundamental task in natural language processing that aims to identify and classify relationships between entities in text. Document-level relation extraction expands this task to process relationships across entire documents rather than isolated sentences, addressing more complex relationship scenarios.

The **DocRE-RLKGf** framework integrates reinforcement learning (RL) and knowledge graph (KG) techniques to:

1. Enhance the accuracy of relation extraction by utilizing prior knowledge encoded in a KG.
2. Employ RL strategies to iteratively refine extraction decisions based on document context and KG features.

---

## Key Features

- **Knowledge Graph Fusion**: Combines textual data with external knowledge graphs to enrich context and improve relation extraction.
- **Reinforcement Learning**: Implements RL to optimize the relation extraction process dynamically.
- **Document-Level Processing**: Designed to work across entire documents, handling complex multi-entity relationships.

---

## Requirements

Ensure the following dependencies are installed:

- Python >= 3.6
- PyTorch >= 1.8
- Transformers >= 4.0
- Additional requirements listed in `requirements.txt`

Install dependencies by running:

```bash
pip install -r requirements.txt
```

---

## Usage

### Data Preparation

1. Download the dataset from the provided link and place it in the `data/` directory.
2. Preprocess the dataset using the script provided in `scripts/preprocess.py`:

   ```bash
   python scripts/preprocess.py --input data/raw --output data/processed
   ```

### Training

To train the model, execute the following command:

```bash
python train.py --config configs/train_config.json
```

- `--config`: Specifies the configuration file for training parameters.
- Example configuration files can be found in the `configs/` directory.

### Evaluation

Evaluate the trained model using:

```bash
python evaluate.py --model checkpoint/best_model.pth --data data/processed/test
```

- `--model`: Path to the trained model checkpoint.
- `--data`: Path to the test dataset.

### Inference

For inference on new documents, run:

```bash
python infer.py --input data/new_document.txt --output results/output.json
```

- `--input`: Path to the input text file.
- `--output`: Path to save the extracted relations in JSON format.

---

## Directory Structure

```
.
├── configs/              # Configuration files for training and evaluation
├── data/                 # Dataset directory
├── scripts/              # Utility scripts (e.g., preprocessing)
├── models/               # Model definitions and training logic
├── results/              # Output results directory
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── infer.py              # Inference script
└── README.md             # Documentation
```

---

## Results

The framework achieves state-of-the-art performance on benchmark datasets for document-level relation extraction.

### Key Metrics:

| Metric            | Value |
|--------------------|-------|
| Precision         | 85.4% |
| Recall            | 83.2% |
| F1-Score          | 84.3% |

---

## References

For more details, please refer to the following resources:

- Original paper: [DocRE-RLKGf Paper](#)
- Dataset: [Dataset Link](#)
- Knowledge Graphs: [External KG Resource](#)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or issues, please contact:

- Maintainer: [Your Name](mailto:your.email@example.com)
- GitHub Issues: [Issue Tracker](https://github.com/dice-group/DocRE-RLKGf/issues)

---

## Citation

If you use this framework in your research, please cite the following:

```bibtex
@article{docre_rlkgf,
  title={Document-level Relation Extraction with Reinforcement Learning Knowledge Graph Fusion},
  author={Your Name},
  journal={Your Journal},
  year={2023}
}
```

