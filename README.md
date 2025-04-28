# Document-level Relation Extraction using Reinforcement Learning with Knowledge Graph Feedback

**DocRE-RLKGf** (Document-level Relation Extraction using Reinforcement Learning with Knowledge Graph feedback) is an approach designed for extracting relations from documents using reinforcement learning and knowledge graph feedback. This method utilizes the strengths of knowledge graphs and reinforcement learning to improve the accuracy of Large Language Models (LLMs) for DocRE.

![architecture](https://github.com/dice-group/DocRE-RLKGf/blob/main/rlkgf.drawio.svg)

## Overview

Relation extraction is a fundamental task in natural language processing that aims to identify and classify relationships between entities in text. Document-level relation extraction expands this task to process relationships across entire documents rather than isolated sentences, addressing more complex relationship scenarios.

The **RLKGf** integrates reinforcement learning (RL) and knowledge graph (KG) in the RLHF paradigm:
1. Enhance the accuracy of relation extraction by utilizing knowledge graphs as a scorer.
2. Employ RL strategies to iteratively refine extraction decisions based on document context and KG feedback.

## Key Features

- **Knowledge Graph based Scoring Function**: Exploit KGs to score the outputs of LLMs.
- **Reinforcement Learning**: Implements RL to optimize the relation extraction process.
- **Document-Level Processing**: Designed to work across entire documents, handling complex multi-entity relationships.

## Requirements

Create the Environment: Use the following command to create the environment from the YAML file.

```bash
conda env create -f environment.yaml
conda activate sparqlgen
```

Additionally, you'll need to install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Data Preparation

1. Download the dataset from the provided link given in the dataset readme file and place it in the `dataset/` directory.
2. Preprocess the dataset using the script provided in `dataset/prompt_data_prep.py`:

```bash
python dataset/prompt_data_prep.py \
  --input_file dataset/docred_train.json \
  --output_file dataset/train_docred_prompt_wES.csv \
  --include_evidence_sentences True
```

### Fine-Tuning LLM

To fine-tune the Large Language Model (LLM) for DocRE using human-annotated data, use the `fine-tune-LLM.ipynb` notebook or run the following command:

```bash
python  fine_tune_llm.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --train_dataset "datasets/train_docred_prompt_wES.csv" \
  --dev_dataset "datasets/dev_docred_prompt_wES.csv" \
  --output_dir "output/fine_tuned_model" \
  --max_steps 1000 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 1
```

Parameters:
- `--model_name`: Pre-trained model to fine-tune (from Hugging Face or local path)
- `--train_dataset`: Path to the training dataset
- `--dev_dataset`: Path to the validation dataset
- `--output_dir`: Directory to save the fine-tuned model
- `--max_steps`: Maximum number of training steps
- `--learning_rate`: Learning rate for optimization
- `--batch_size`: Batch size for training
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients

### Generate Outputs

To generate outputs for the DS (Distant Supervision) version, use the `generate_output.ipynb` notebook or run the following command:

```bash
python  generate_outputs.py \
  --model_path "output/fine_tuned_model" \
  --dataset_path "datasets/ds_docred_prompt_wES.csv" \
  --output_path "output/generated_outputs.csv" \
  --num_outputs 3 \
  --temperature 0.3 \
  --top_k 5 \
  --top_p 0.9 \
  --repetition_penalty 1.2 \
  --max_new_tokens 512
```

Parameters:
- `--model_path`: Path to the fine-tuned model
- `--dataset_path`: Path to the dataset to generate outputs for
- `--output_path`: Path to save the generated outputs
- `--num_outputs`: Number of outputs to generate per document (default: 3)
- `--temperature`: Temperature for sampling (default: 0.3)
- `--top_k`: Top-k sampling parameter (default: 5)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--repetition_penalty`: Penalty for repetition (default: 1.2)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)

### Scoring Triples

To score the extracted triples using knowledge graph feedback, execute the following command:

```bash
python calculate_triple_scores.py \
  --predictions_file "output/generated_outputs.csv" \
  --sentences_file "datasets/sentences.csv" \
  --output_file "output/scored_predictions.csv" \
  --confidence 0.5
```

Parameters:
- `--predictions_file`: Path to the file containing extracted triples
- `--sentences_file`: Path to the file containing evidence sentences
- `--output_file`: Path to save the scored predictions
- `--confidence`: Confidence parameter for scoring (default: 0.5)

### Training DPO (Direct Preference Optimization)

To train the model using DPO based on the scored outputs, run the following command:

```bash
python dpo.py \
  --dataset_path "output/dpo_scored.csv" \
  --model_checkpoint "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --output_dir "output/dpo_model" \
  --max_steps 1000 \
  --logging_steps 1 \
  --save_steps 500 \
  --learning_rate 1e-4 \
  --beta 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1
```

Parameters:
- `--dataset_path`: Path to the dataset with chosen and rejected outputs
- `--model_checkpoint`: Model checkpoint to use
- `--output_dir`: Directory to save the trained model
- `--max_steps`: Maximum training steps
- `--logging_steps`: Logging steps
- `--save_steps`: Save model every N steps
- `--learning_rate`: Learning rate
- `--beta`: Beta parameter for KL penalty in DPO
- `--per_device_train_batch_size`: Batch size per device for training
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients

### Training Reward Model (Optional for PPO)

If you choose to run PPO (Proximal Policy Optimization) instead of DPO, you need to train a reward model first. Use the `reward_model_training.ipynb` notebook or run the following command:

```bash
python  train_reward_model.py \
  --model_name "gpt2" \
  --dataset_path "datasets/rewardModelTrain.csv" \
  --output_dir "output/reward_model" \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1
```

Parameters:
- `--model_name`: Base model for the reward model
- `--dataset_path`: Path to the dataset for training the reward model
- `--output_dir`: Directory to save the trained reward model
- `--num_train_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--per_device_train_batch_size`: Batch size per device for training
- `--per_device_eval_batch_size`: Batch size per device for evaluation

### Training with PPO (Alternative to DPO)

To train the model using PPO with the trained reward model, run the following command:

```bash
python  ppo_trainer.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --reward_model_name "output/reward_model" \
  --dataset_path "datasets/ppo_train.csv" \
  --output_dir "output/ppo_model" \
  --learning_rate 1.41e-5 \
  --max_steps 1000 \
  --batch_size 4 \
  --mini_batch_size 1 \
  --gradient_accumulation_steps 1
```

Parameters:
- `--model_name`: Name or path of the pretrained model
- `--reward_model_name`: Name or path of the reward model
- `--dataset_path`: Path to the dataset CSV file
- `--output_dir`: Directory to save the trained model
- `--learning_rate`: Learning rate for optimization
- `--max_steps`: Maximum number of training steps
- `--batch_size`: Batch size for training
- `--mini_batch_size`: Mini-batch size for PPO updates
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients

### Inference

For inference on new documents, use the fine-tuned model or the model trained with DPO/PPO:

```bash
python  generate_outputs.py \
  --model_path "output/dpo_model/final_checkpoint" \
  --dataset_path "datasets/test_docred_prompt_wES.csv" \
  --output_path "output/test_predictions.csv" \
  --num_outputs 1 \
  --temperature 0.3
```

## Hyperparameters

### Fine-tuning Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Learning rate for optimization |
| Batch Size | 1 | Batch size per device |
| Gradient Accumulation Steps | 1 | Number of steps to accumulate gradients |
| Max Steps | 1000 | Maximum number of training steps |
| LR Scheduler | cosine | Learning rate scheduler type |
| Warmup Steps | 100 | Number of warmup steps |
| Weight Decay | 0.05 | Weight decay for regularization |
| LoRA r | 16 | Rank of the update matrices |
| LoRA alpha | 32 | Alpha parameter for LoRA scaling |
| LoRA dropout | 0.05 | Dropout probability for LoRA layers |

### DPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Learning rate for optimization |
| Batch Size | 1 | Batch size per device |
| Gradient Accumulation Steps | 1 | Number of steps to accumulate gradients |
| Max Steps | 1000 | Maximum number of training steps |
| Beta | 0.1 | Beta parameter for KL penalty |
| LR Scheduler | cosine | Learning rate scheduler type |
| Warmup Steps | 100 | Number of warmup steps |
| LoRA r | 16 | Rank of the update matrices |
| LoRA alpha | 32 | Alpha parameter for LoRA scaling |
| LoRA dropout | 0.05 | Dropout probability for LoRA layers |

### PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1.41e-5 | Learning rate for optimization |
| Batch Size | 4 | Batch size for training |
| Mini Batch Size | 1 | Mini-batch size for PPO updates |
| Gradient Accumulation Steps | 1 | Number of steps to accumulate gradients |
| KL Penalty | 0.1 | Target KL divergence |
| Max Steps | 1000 | Maximum number of training steps |
| Response Length | 256-512 | Range of response lengths |

### Reward Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2e-5 | Learning rate for optimization |
| Batch Size | 4 | Batch size per device |
| Epochs | 3 | Number of training epochs |
| Weight Decay | 0.001 | Weight decay for regularization |
| LoRA r | 8 | Rank of the update matrices |
| LoRA alpha | 32 | Alpha parameter for LoRA scaling |
| LoRA dropout | 0.1 | Dropout probability for LoRA layers |

## Prompt Templates

### Document Relation Extraction Prompt

```
You are an expert in extracting relations from documents. Given a document, identify all entity pairs and the relations between them.

Document:
{document_text}

Instructions:
1. Identify all entities in the document.
2. For each pair of entities that have a relation, extract:
   - Head Entity: The first entity in the relation
   - Tail Entity: The second entity in the relation
   - Relation: The type of relation between the entities
   - Evidence Sentences: The sentence IDs that support this relation

Format your answer as follows:
Head Entity: [entity name]
Tail Entity: [entity name]
Relation: [relation type]
Evidence Sentences: [list of sentence IDs]

Example Output:
```


## Results

RLKGf achieves state-of-the-art performance on benchmark datasets for document-level relation extraction.

### Key Metrics:

| Dataset            | F1    |
|--------------------|-------|
| DocRED             | 75.5% |
| Re-DocRED          | 81.3% |

## References

For more details, please refer to the following resources:
- Original paper: [DocRE-RLKGf Paper](#)

## License

The source code of this repo is published under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Contact

For questions or issues, please contact:
Due to anonymous submission the information will be provided after reviews.
- Maintainer:  
- GitHub Issues:

## Citation

If you use this framework in your research, please cite the following:

```bibtex
@article{docre_rlkgf,
  title={RLKGf: Reinforcement Learning using Knowledge Graph Feedback for Document-level Relation Extraction},
  author={-},
  journal={-},
  year={2025}
}
```
