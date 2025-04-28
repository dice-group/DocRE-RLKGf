import argparse
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Load evaluation metric
accuracy = evaluate.load("accuracy")

class RewardModelTrainer:
    """
    Trainer class for reward model training in document-level relation extraction.
    This class handles the training of a reward model that can score the quality of extracted relations.
    """
    
    def __init__(
        self,
        model_name,
        dataset_path,
        output_dir,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        weight_decay=0.001,
        max_length=512,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1
    ):
        """
        Initialize the reward model trainer.
        
        Args:
            model_name: Name or path of the base model
            dataset_path: Path to the dataset CSV file
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            per_device_train_batch_size: Batch size per device for training
            per_device_eval_batch_size: Batch size per device for evaluation
            weight_decay: Weight decay for regularization
            max_length: Maximum sequence length
            lora_r: Rank parameter for LoRA
            lora_alpha: Alpha parameter for LoRA
            lora_dropout: Dropout probability for LoRA
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Initialize tokenizer and model
        self.setup_tokenizer_and_model()
        
        # Load and prepare dataset
        self.load_dataset()
    
    def setup_tokenizer_and_model(self):
        """Set up the tokenizer and model for reward modeling."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set EOS token as padding token
        
        # Define the PEFT configuration for LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )
        
        # Load the model and apply PEFT (LoRA)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=1, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Set padding token in model configuration
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = False  # Disable caching for training
    
    def load_dataset(self):
        """Load and prepare the dataset for reward model training."""
        # Load dataset
        train_df = pd.read_csv(self.dataset_path)
        dataset = Dataset.from_pandas(train_df)
        
        # Split the dataset into training and validation
        train_val_split = dataset.train_test_split(test_size=0.1)
        self.train_dataset = train_val_split['train']
        self.val_dataset = train_val_split['test']
        
        print(f"Loaded dataset with {len(self.train_dataset)} training examples and {len(self.val_dataset)} validation examples")
        
        # Preprocess datasets
        self.preprocess_datasets()
    
    def preprocess_datasets(self):
        """Preprocess the datasets for reward model training."""
        # Define preprocessing function
        def preprocess_function(examples):
            new_examples = {
                "input_ids_j": [],
                "attention_mask_j": [],
                "input_ids_k": [],
                "attention_mask_k": [],
            }
            
            # Process each example
            for question, response_j, response_k in zip(
                examples["question"], 
                examples["response_j"], 
                examples["response_k"]
            ):
                # Format inputs
                tokenized_j = self.tokenizer(
                    "Question: " + question + "\n\nAnswer: " + str(response_j), 
                    truncation=True,
                    max_length=self.max_length
                )
                tokenized_k = self.tokenizer(
                    "Question: " + question + "\n\nAnswer: " + str(response_k), 
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Store tokenized inputs
                new_examples["input_ids_j"].append(tokenized_j["input_ids"])
                new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
                new_examples["input_ids_k"].append(tokenized_k["input_ids"])
                new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
            
            return new_examples
        
        # Apply preprocessing
        num_proc = min(os.cpu_count(), 24)  # Use multiple processes but not more than 24
        
        self.processed_train_dataset = self.train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=self.train_dataset.column_names,
        )
        
        # Filter examples that are too long
        self.processed_train_dataset = self.processed_train_dataset.filter(
            lambda x: len(x["input_ids_j"]) <= self.max_length and len(x["input_ids_k"]) <= self.max_length
        )
        
        self.processed_val_dataset = self.val_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=self.val_dataset.column_names,
        )
        
        print(f"Processed {len(self.processed_train_dataset)} training examples and {len(self.processed_val_dataset)} validation examples")
    
    def create_data_collator(self):
        """Create a data collator for reward model training."""
        @dataclass
        class RewardDataCollatorWithPadding:
            tokenizer: PreTrainedTokenizerBase
            padding: Union[bool, str] = True
            max_length: Optional[int] = self.max_length
            pad_to_multiple_of: Optional[int] = None
            return_tensors: str = "pt"
            
            def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
                features_j = []
                features_k = []
                
                for feature in features:
                    features_j.append(
                        {
                            "input_ids": feature["input_ids_j"],
                            "attention_mask": feature["attention_mask_j"],
                        }
                    )
                    features_k.append(
                        {
                            "input_ids": feature["input_ids_k"],
                            "attention_mask": feature["attention_mask_k"],
                        }
                    )
                
                batch_j = self.tokenizer.pad(
                    features_j,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                
                batch_k = self.tokenizer.pad(
                    features_k,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                
                batch = {
                    "input_ids_j": batch_j["input_ids"],
                    "attention_mask_j": batch_j["attention_mask"],
                    "input_ids_k": batch_k["input_ids"],
                    "attention_mask_k": batch_k["attention_mask"],
                    "return_loss": True,
                }
                
                return batch
        
        return RewardDataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def create_reward_trainer(self):
        """Create a custom trainer for reward model training."""
        class RewardTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                rewards_j = model(
                    input_ids=inputs["input_ids_j"], 
                    attention_mask=inputs["attention_mask_j"]
                )[0]
                
                rewards_k = model(
                    input_ids=inputs["input_ids_k"], 
                    attention_mask=inputs["attention_mask_k"]
                )[0]
                
                loss = -torch.nn.functional.logsigmoid(rewards_j - rewards_k).mean()
                
                if return_outputs:
                    return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
                
                return loss
        
        return RewardTrainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for the reward model."""
        predictions, _ = eval_pred
        # Here, predictions is rewards_j and rewards_k.
        # We want to see how much of the time rewards_j > rewards_k.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)
    
    def train(self):
        """Train the reward model."""
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            deepspeed=None,
            local_rank=-1,
            remove_unused_columns=False,  # Important for custom inputs
            bf16=torch.cuda.is_available(),
            logging_strategy="steps",
            logging_steps=10,
            optim="adamw_hf",
            lr_scheduler_type="linear",
        )
        
        # Create data collator
        data_collator = self.create_data_collator()
        
        # Create reward trainer class
        RewardTrainer = self.create_reward_trainer()
        
        # Initialize the trainer
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.processed_train_dataset,
            eval_dataset=self.processed_val_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting reward model training...")
        trainer.train()
        
        # Save the trained model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def main():
    """Main function to run reward model training."""
    parser = argparse.ArgumentParser(description="Train a reward model for document-level relation extraction.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name or path of the base model")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the trained model")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate for optimization")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, 
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, 
                        help="Batch size per device for evaluation")
    parser.add_argument("--weight_decay", type=float, default=0.001, 
                        help="Weight decay for regularization")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="Rank parameter for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1, 
                        help="Dropout probability for LoRA")
    
    args = parser.parse_args()
    
    # Initialize and train reward model
    trainer = RewardModelTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    trainer.train()
    
    print("Reward model training completed successfully!")

if __name__ == "__main__":
    import os
    main()
