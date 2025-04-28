import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
import argparse

class DocREGenerator:
    """
    Generator class for Document Relation Extraction using fine-tuned LLMs.
    This class handles loading models and generating outputs for DocRE tasks.
    """
    
    def __init__(
        self,
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.3,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=512
    ):
        """
        Initialize the DocRE generator.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run the model on ('cuda' or 'cpu')
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_path = model_path
        self.device = device
        
        # Generation parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Create generation config
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
    
    def format_prompt(self, document):
        """
        Format the document into a prompt for the model.
        
        Args:
            document: Document text
            
        Returns:
            Formatted prompt
        """
        return f"{document}\nExample Output"
    
    def generate_output(self, document):
        """
        Generate relation extraction output for a document.
        
        Args:
            document: Document text
            
        Returns:
            Generated relation extraction output
        """
        prompt = self.format_prompt(document)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate output
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        
        # Decode and extract the generated part
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Split to get only the generated part after "Example Output"
        if "Example Output" in result:
            result = result.split("Example Output")[1].strip()
        
        return result
    
    def generate_multiple_outputs(self, document, num_outputs=3):
        """
        Generate multiple relation extraction outputs for a document.
        
        Args:
            document: Document text
            num_outputs: Number of outputs to generate
            
        Returns:
            List of generated outputs
        """
        outputs = []
        for _ in range(num_outputs):
            output = self.generate_output(document)
            outputs.append(output)
        
        return outputs
    
    def process_dataset(self, dataset_path, output_path, num_outputs=3):
        """
        Process a dataset and generate outputs for each document.
        
        Args:
            dataset_path: Path to the dataset CSV file
            output_path: Path to save the outputs
            num_outputs: Number of outputs to generate per document
            
        Returns:
            DataFrame with original documents and generated outputs
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Generate outputs for each document
        output_columns = [f"output{i+1}" for i in range(num_outputs)]
        
        for i, row in df.iterrows():
            document = row["prompt"]
            outputs = self.generate_multiple_outputs(document, num_outputs)
            
            for j, output in enumerate(outputs):
                df.loc[i, output_columns[j]] = output
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(df)} documents")
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        return df

def main():
    """Main function to run the generator."""
    parser = argparse.ArgumentParser(description="Generate DocRE outputs using a fine-tuned model.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the fine-tuned model")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset CSV file")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the outputs")
    parser.add_argument("--num_outputs", type=int, default=3, 
                        help="Number of outputs to generate per document")
    parser.add_argument("--temperature", type=float, default=0.3, 
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, 
                        help="Penalty for repetition")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DocREGenerator(
        model_path=args.model_path,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens
    )
    
    # Process dataset
    generator.process_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_outputs=args.num_outputs
    )

if __name__ == "__main__":
    main()
