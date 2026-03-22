#!/bin/bash
# Patch Embedding Ablation Experiments (Figure 3)
# Cross-granularity federated learning: ETTh (1-hour) + ETTm (15-minute)

# ETT1 (ETTh1 + ETTm1)
python run_ablation_patch.py --ett_version 1 --pred_len 96 --d_model 128 --en_d_ff 512 --de_d_ff 512
python run_ablation_patch.py --ett_version 1 --pred_len 192 --d_model 128 --en_d_ff 512 --de_d_ff 512
python run_ablation_patch.py --ett_version 1 --pred_len 336 --d_model 128 --en_d_ff 512 --de_d_ff 512
python run_ablation_patch.py --ett_version 1 --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512

# ETT2 (ETTh2 + ETTm2)
python run_ablation_patch.py --ett_version 2 --pred_len 96 --d_model 128 --en_d_ff 512 --de_d_ff 512
python run_ablation_patch.py --ett_version 2 --pred_len 192 --d_model 128 --en_d_ff 512 --de_d_ff 512
python run_ablation_patch.py --ett_version 2 --pred_len 336 --d_model 128 --en_d_ff 512 --de_d_ff 512
python run_ablation_patch.py --ett_version 2 --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512
