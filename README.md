# DocRE-RLKGf
This is the repo for our upcoming paper

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
