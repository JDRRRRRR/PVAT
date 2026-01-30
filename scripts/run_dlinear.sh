#!/bin/bash
# DLinear Model Experiments
# Simple Decomposition-based Linear Model for Time-Series Forecasting

# ETTh1 Dataset
python run.py --model DLinear --data ETTh1 --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data ETTh1 --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data ETTh1 --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data ETTh1 --features M --target OT --seq_len 96 --pred_len 720

# ETTh2 Dataset
python run.py --model DLinear --data ETTh2 --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data ETTh2 --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data ETTh2 --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data ETTh2 --features M --target OT --seq_len 96 --pred_len 720

# ETTm1 Dataset
python run.py --model DLinear --data ETTm1 --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data ETTm1 --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data ETTm1 --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data ETTm1 --features M --target OT --seq_len 96 --pred_len 720

# ETTm2 Dataset
python run.py --model DLinear --data ETTm2 --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data ETTm2 --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data ETTm2 --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data ETTm2 --features M --target OT --seq_len 96 --pred_len 720

# Electricity Dataset
python run.py --model DLinear --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 720

# Exchange Rate Dataset
python run.py --model DLinear --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 720

# Traffic Dataset
python run.py --model DLinear --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 720

# Weather Dataset
python run.py --model DLinear --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 96
python run.py --model DLinear --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 192
python run.py --model DLinear --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 336
python run.py --model DLinear --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 720
