# RLKGf

**DocRE-RLKGf** (Document-level Relation Extraction with Reinforcement Learning using Knowledge Graph feedback) is an approach designed for extracting relations from documents using reinforcement learning and knowledge graph. This method utilizes the strengths of knowledge graphs and reinforcement learning to improve the accuracy of Large Langauge Models (LLMs) for DocRE.


## Overview

Relation extraction is a fundamental task in natural language processing that aims to identify and classify relationships between entities in text. Document-level relation extraction expands this task to process relationships across entire documents rather than isolated sentences, addressing more complex relationship scenarios.

The **RLKGf** integrates reinforcement learning (RL) and knowledge graph (KG) in the RLHF paradigm:

1. Enhance the accuracy of relation extraction by utilizing knowledge graphs is a scorer.
2. Employ RL strategies to iteratively refine extraction decisions based on document context and KG feadback.


## Key Features

- **Knowledge Graph based Scoring Function**: Exploit KGs to score the outputs of LLMs.
- **Reinforcement Learning**: Implements RL to optimize the relation extraction process.
- **Document-Level Processing**: Designed to work across entire documents, handling complex multi-entity relationships.

![architecture](https://github.com/dice-group/DocRE-RLKGf/blob/main/rlkgf.drawio.svg)

## Requirements

Ensure the following dependencies are installed:

- Python >= 3.9
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

1. Download the dataset from the provided link given in the dataset readme file and place it in the `dataset/` directory.
2. Preprocess the dataset using the script provided in `dataset/prompt_data_prep.py`:


### Fine-Tuning LLM

To fine-tune the Large Language Model (LLM) for DocRE using human-annotated data, use the `fine-tune-LLM.ipynb` notebook. Please ensure the following adjustments are made before running the notebook:

1. Provide the link to the pre-trained model. If using a model from Hugging Face, include your access token for authentication.

2. Update the OUTPUT_DIRECTORY variable to specify the directory where the results and fine-tuned model will be saved.

3. Prepare a prompt-based version of the human-annotated DocRED dataset using the dataset preparation script provided in the repository.

4. Set an appropriate seq_length value for the dataset. Adjust other training parameters such as batch size, learning rate, and epochs to suit your specific requirements (optional).

### Generate Output

To generate the output for the DS (Distant Supervision) version, use the ```generate_output.ipynb``` notebook. Please ensure the following adjustments:

1. Provide the model you saved during fine-tuning.

2. Provide the prompt-based version of the distant supervision dataset.

3. Save the output in CSV format.

### Scoring Prompts

To score the extracted triples, execute the following command:

```bash
python calculate_triple_scores.py --triples_file triples.csv --sentences_file sentences.csv
```

### Training DPO

To train the DPO (Direct Preference Optimization), run the following command:
```bash
python dpo.py --dataset_path datasets/dpo_scored.csv \
  --model_checkpoint meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir ./RL_final_checkpoints \
  --max_steps 1000 \
  --logging_steps 1 \
  --save_steps 500 \
  --learning_rate 1e-4
``` 
### Training Reward Model (Optional)

If you choose to run PPO (Proximal Policy Optimization) instead of DPO, you need to train a reward model first. Use the ```reward_model_training.ipynb``` notebook. The code is self-explanatory, and you can adjust the parameters according to your needs.

### Evaluation

Evaluate the trained model using:

```bash
python evaluate.py --model checkpoint/best_model.pth --data data/processed/test
```

- `--model`: Path to the trained model checkpoint.
- `--data`: Path to the test dataset.

### Inference

For inference on new documents, run the same code of output generation but provide the final model after running DPO instead of using Fine-tuned model:

```bash
generate_output.ipynb
```


---

## Results

RLKGf achieves state-of-the-art performance on benchmark datasets for document-level relation extraction.

### Key Metrics:

| Dataset            | F1 |
|--------------------|-------|
| DocRED         | 71.3% |
| Re-DocRED            | 79.1% |
| DWIE         | 83.3% |

---

## References

For more details, please refer to the following resources:

- Original paper: [DocRE-RLKGf Paper](#)

---

## License

The source code of this repo is published under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

---

## Contact

For questions or issues, please contact:

- Maintainer: [Email](Will give info soon) 
- GitHub Issues: [Issue Tracker](Under review)

---

## Citation

If you use this framework in your research, please cite the following:

```bibtex
@article{docre_rlkgf,
  title={RLKGf: Reinforcement Learning using Knowledge Graph Feedback for Document-level Relation Extraction},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

