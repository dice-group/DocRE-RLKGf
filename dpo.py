import os
import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

def load_and_prepare_data(dataset_path):
    """Load and preprocess the dataset."""
    result_df = pd.read_csv(dataset_path)
    result_df = result_df.rename(columns={
        'highest_score_output': 'chosen',
        'lowest_score_output': 'rejected'
    })
    dataset = Dataset.from_pandas(result_df)

    # Filter samples to ensure they fit within a token limit
    train_dataset = dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 8096
        and len(x["prompt"]) + len(x["rejected"]) <= 8096,
        num_proc=1,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering. Please adjust the filtering conditions.")
    
    return train_dataset

def initialize_model(model_checkpoint, torch_dtype):
    """Load the pre-trained model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=True,
        device_map={"": torch.cuda.current_device()}  # Use the current GPU
    )
    model.config.use_cache = False  # Disable caching for training
    return model

def train_model(train_dataset, model, tokenizer, args):
    """Train the model using DPO."""
    eval_dataset = train_dataset.shuffle(seed=42).select(range(len(train_dataset) // 10))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "out_proj",
            "fc_in", "fc_out", "wte"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Reference model is optional
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Ensure tokenizer is passed
        peft_config=peft_config,
    )
    
    # Train the model
    dpo_trainer.train()

    # Save the model
    dpo_trainer.save_model()

    # Save the final checkpoint
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using DPO.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input dataset CSV file.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Model checkpoint to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model every N steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()

def main():
    """Main function to orchestrate training."""
    args = parse_arguments()

    # Load and prepare data
    train_dataset = load_and_prepare_data(args.dataset_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token

    # Initialize model
    torch_dtype = torch.float
    model = initialize_model(args.model_checkpoint, torch_dtype)

    # Define training arguments
    training_args = DPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        output_dir=args.output_dir,
        report_to='wandb',
        lr_scheduler_type='cosine',
        warmup_steps=100,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_training",
        seed=0
    )

    # Train the model
    train_model(train_dataset, model, tokenizer, training_args)

if __name__ == "__main__":
    main()

