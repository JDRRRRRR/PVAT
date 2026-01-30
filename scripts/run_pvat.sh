#!/bin/bash
# PVAT Model Experiments
# Patch- and Variable-Aligned Transformer for Time-Series Forecasting

# ETTh1 Dataset
python run.py --model PVAT --data ETTh1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTh1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTh1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTh1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1

# ETTh2 Dataset
python run.py --model PVAT --data ETTh2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTh2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTh2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTh2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

# ETTm1 Dataset
python run.py --model PVAT --data ETTm1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTm1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTm1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTm1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

# ETTm2 Dataset
python run.py --model PVAT --data ETTm2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTm2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTm2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run.py --model PVAT --data ETTm2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1

# Electricity Dataset
python run.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2

# Exchange Rate Dataset
python run.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1

# Traffic Dataset
python run.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

# Weather Dataset
python run.py --model PVAT --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run.py --model PVAT --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
