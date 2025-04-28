import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import pandas as pd
from datasets import Dataset

class PPODocRETrainer:
    """
    Trainer class for Proximal Policy Optimization (PPO) on Document Relation Extraction.
    This implements the PPO algorithm for fine-tuning language models with KG feedback.
    """
    
    def __init__(
        self,
        model_name,
        reward_model_name,
        dataset_path,
        output_dir,
        learning_rate=1.41e-5,
        max_steps=1000,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=1
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            model_name: Name or path of the pretrained model
            reward_model_name: Name or path of the reward model
            dataset_path: Path to the dataset CSV file
            output_dir: Directory to save the trained model
            learning_rate: Learning rate for optimization
            max_steps: Maximum number of training steps
            batch_size: Batch size for training
            mini_batch_size: Mini-batch size for PPO updates
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model_name = model_name
        self.reward_model_name = reward_model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Initialize reward model
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            reward_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Configure PPO
        self.ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            kl_penalty="kl",
            seed=42,
            log_with="wandb",
            max_steps=max_steps
        )
        
        # Load dataset
        self.load_dataset()
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset
        )
        
        # Response length sampler
        self.response_length_sampler = LengthSampler(256, 512)
    
    def load_dataset(self):
        """Load and prepare the dataset for PPO training."""
        df = pd.read_csv(self.dataset_path)
        self.dataset = Dataset.from_pandas(df)
        print(f"Loaded dataset with {len(self.dataset)} examples")
    
    def compute_reward(self, prompt, response):
        """
        Compute reward for a given response using the reward model.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Reward value
        """
        inputs = self.tokenizer(f"{prompt}\n{response}", return_tensors="pt").to(self.reward_model.device)
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            reward = outputs.logits[0, -1].item()  # Use the last token's logit as reward
        return reward
    
    def generate_response(self, prompt):
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        response_length = self.response_length_sampler()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        generation_output = self.model.generate(
            **inputs,
            max_new_tokens=response_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response = self.tokenizer.decode(generation_output.sequences[0][inputs.input_ids.shape[1]:])
        return response
    
    def train_step(self, batch):
        """
        Perform a single PPO training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of training statistics
        """
        # Get prompts from batch
        prompts = batch["prompt"]
        
        # Generate responses
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt)
            responses.append(response)
        
        # Tokenize prompts and responses
        prompt_tensors = [
            self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            for prompt in prompts
        ]
        
        response_tensors = [
            self.tokenizer(response, return_tensors="pt").input_ids.to(self.model.device)
            for response in responses
        ]
        
        # Compute rewards
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.compute_reward(prompt, response)
            rewards.append(reward)
        
        # Run PPO step
        stats = self.ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        return stats
    
    def train(self):
        """Train the model using PPO."""
        for step, batch in enumerate(self.ppo_trainer.dataloader):
            if step >= self.ppo_config.max_steps:
                break
            
            # Perform training step
            stats = self.train_step(batch)
            
            # Log statistics
            print(f"Step {step}: {stats}")
            
            # Save checkpoint
            if step > 0 and step % 100 == 0:
                self.save_model(f"checkpoint-{step}")
        
        # Save final model
        self.save_model("final")
    
    def save_model(self, suffix):
        """
        Save the model checkpoint.
        
        Args:
            suffix: Suffix to add to the checkpoint directory name
        """
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{suffix}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Model saved to {checkpoint_dir}")

def main():
    """Main function to run PPO training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model using PPO for DocRE.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name or path of the pretrained model")
    parser.add_argument("--reward_model_name", type=str, required=True, 
                        help="Name or path of the reward model")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the trained model")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, 
                        help="Learning rate for optimization")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=1, 
                        help="Mini-batch size for PPO updates")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of steps to accumulate gradients")
    
    args = parser.parse_args()
    
    # Initialize and train
    trainer = PPODocRETrainer(
        model_name=args.model_name,
        reward_model_name=args.reward_model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
