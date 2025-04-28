import os
import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

def load_and_prepare_data(dataset_path):
    """
    Load and preprocess the dataset for DPO training.
    
    Args:
        dataset_path: Path to CSV file containing prompts, chosen and rejected outputs
        
    Returns:
        Dataset object ready for DPO training
    """
    # Load the dataset
    result_df = pd.read_csv(dataset_path)
    
    # Rename columns to match DPO requirements
    result_df = result_df.rename(columns={
        'highest_score_output': 'chosen',
        'lowest_score_output': 'rejected'
    })
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(result_df)

    # Filter samples to ensure they fit within token limit
    # This is important to prevent OOM errors during training
    train_dataset = dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 8096
        and len(x["prompt"]) + len(x["rejected"]) <= 8096,
        num_proc=1,
    )
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering. Please adjust the filtering conditions.")
    
    return train_dataset

def initialize_model(model_checkpoint, torch_dtype):
    """
    Load the pre-trained model for DPO fine-tuning.
    
    Args:
        model_checkpoint: Path or name of the pre-trained model
        torch_dtype: PyTorch data type for model weights
        
    Returns:
        Loaded model ready for DPO training
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        device_map={"": torch.cuda.current_device()}  # Use the current GPU
    )
    
    # Disable caching for training to save memory
    model.config.use_cache = False
    
    return model

def train_model(train_dataset, model, tokenizer, args):
    """
    Train the model using Direct Preference Optimization (DPO).
    
    Args:
        train_dataset: Dataset object containing training data
        model: Pre-trained model to fine-tune
        tokenizer: Tokenizer for the model
        args: Training arguments
        
    Returns:
        None (saves the model to disk)
    """
    # Create a small evaluation dataset
    eval_dataset = train_dataset.shuffle(seed=42).select(range(len(train_dataset) // 10))

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=16,  # Rank of the update matrices
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        target_modules=[
            "q_proj", "v_proj", "k_proj", "out_proj",
            "fc_in", "fc_out", "wte"
        ],  # Which modules to apply LoRA to
        bias="none",  # Don't train bias parameters
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )

    # Initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Reference model is optional, will use initial weights if None
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    
    # Train the model
    dpo_trainer.train()

    # Save the model
    dpo_trainer.save_model()

    # Save the final checkpoint
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

def parse_arguments():
    """
    Parse command-line arguments for DPO training.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a model using Direct Preference Optimization (DPO).")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the input dataset CSV file with prompts, chosen and rejected outputs.")
    parser.add_argument("--model_checkpoint", type=str, required=True, 
                        help="Model checkpoint to use (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct').")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the trained model.")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="Maximum training steps.")
    parser.add_argument("--logging_steps", type=int, default=1, 
                        help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Save model every N steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.1, 
                        help="Beta parameter for KL penalty in DPO.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                        help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of steps to accumulate gradients before updating weights.")
    return parser.parse_args()

def main():
    """Main function to orchestrate DPO training."""
    args = parse_arguments()

    # Load and prepare data
    train_dataset = load_and_prepare_data(args.dataset_path)
    print(f"Loaded dataset with {len(train_dataset)} examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token

    # Initialize model
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = initialize_model(args.model_checkpoint, torch_dtype)

    # Define training arguments
    training_args = DPOConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        output_dir=args.output_dir,
        report_to='wandb',  # Report training metrics to Weights & Biases
        lr_scheduler_type='cosine',  # Use cosine learning rate scheduler
        warmup_steps=100,  # Number of warmup steps for learning rate scheduler
        bf16=True if torch.cuda.is_available() else False,  # Use bfloat16 precision if available
        remove_unused_columns=False,  # Keep all columns in the dataset
        run_name="dpo_training",  # Run name for logging
        seed=42,  # Random seed for reproducibility
        beta=args.beta,  # Beta parameter for KL penalty in DPO
    )

    # Train the model
    train_model(train_dataset, model, tokenizer, training_args)
    
    print("DPO training completed successfully!")

if __name__ == "__main__":
    main()
