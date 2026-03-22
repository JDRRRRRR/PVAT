#!/bin/bash
# VE Table Ablation Experiments (Figure 4)
# Federated learning with heterogeneous variable sets

# Electricity (321 variables)
python run_ablation_ve.py --data electricity --root_path ./dataset/electricity/ --pred_len 96 --d_model 256 --en_d_ff 1024 --de_d_ff 1024

# Traffic (862 variables)
python run_ablation_ve.py --data traffic --root_path ./dataset/traffic/ --pred_len 96 --d_model 256 --en_d_ff 1024 --de_d_ff 1024
