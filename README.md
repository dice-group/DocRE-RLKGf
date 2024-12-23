# DocRE-RLKGf
This repo contains code for our RLKGf for DocRE

![architecture](https://github.com/dice-group/DocRE-RLKGf/blob/main/rlkgf%20(6).pdf)

### Fine Tuning LLMs
To fine tune the the Large language model for DocRE using human annotated data run the `fine-tune-LLM.ipynb` To properly run the code please correct the follwoing: 
1. Provide link of the model (If you provide link from Hugginface please also use your access token)
2. Replace the OUTPUT_DIRECTORY to your own directory.
3. Please provide the prompt version of human annotated DocRED dataset (You can create the prompt version using our dataset code)
4. Provide the seq_length value to the dataset and also adjust the training paramaeter according to particular need. (optional)

### Generate output
To generate the output for the DS version please use the following code:
`generate_output.ipynb`
1. Provide the model you saved while fine-tuning
2. Provide the prompt version of the distant supervsision dataset
3. Save the output in the csv format


### Scoring Prompts
The following line of code will start scoring the triples:

`python calculate_triple_scores.py --triples_file triples.csv --sentences_file sentences.csv`


### Training DPO
To train the dpo use the folloing line of code

`python dpo.py --dataset_path datasets/dpo_scored.csv \
                    --model_checkpoint meta-llama/Meta-Llama-3.1-8B-Instruct \
                    --output_dir ./RL_final_checkpoints \
                    --max_steps 1000 \
                    --logging_steps 1 \
                    --save_steps 500 \
                    --learning_rate 1e-4`
