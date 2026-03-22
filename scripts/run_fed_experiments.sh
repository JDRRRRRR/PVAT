#!/bin/bash
# Federated Learning Experiments for PVAT
# Table 2: Federated multivariate-to-univariate forecasting

# ETTh1
python run_fed.py --model PVAT --data ETTh1 --features MS --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTh1 --features MS --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTh1 --features MS --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTh1 --features MS --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1

# ETTh2
python run_fed.py --model PVAT --data ETTh2 --features MS --pred_len 96 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTh2 --features MS --pred_len 192 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTh2 --features MS --pred_len 336 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTh2 --features MS --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

# ETTm1
python run_fed.py --model PVAT --data ETTm1 --features MS --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTm1 --features MS --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTm1 --features MS --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTm1 --features MS --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

# ETTm2
python run_fed.py --model PVAT --data ETTm2 --features MS --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTm2 --features MS --pred_len 192 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTm2 --features MS --pred_len 336 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data ETTm2 --features MS --pred_len 720 --d_model 256 --en_d_ff 1024 --de_d_ff 1024 --en_layers 1 --de_layers 1

# Electricity
python run_fed.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features MS --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run_fed.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features MS --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run_fed.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features MS --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2
python run_fed.py --model PVAT --data electricity --root_path ./dataset/electricity/ --features MS --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 2 --de_layers 2

# Exchange Rate
python run_fed.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features MS --pred_len 96 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features MS --pred_len 192 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features MS --pred_len 336 --d_model 64 --en_d_ff 256 --de_d_ff 256 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data exchange_rate --root_path ./dataset/exchange_rate/ --features MS --pred_len 720 --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1

# Traffic
python run_fed.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features MS --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features MS --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features MS --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data traffic --root_path ./dataset/traffic/ --features MS --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1

# Weather
python run_fed.py --model PVAT --data weather --root_path ./dataset/weather/ --features MS --pred_len 96 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data weather --root_path ./dataset/weather/ --features MS --pred_len 192 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data weather --root_path ./dataset/weather/ --features MS --pred_len 336 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
python run_fed.py --model PVAT --data weather --root_path ./dataset/weather/ --features MS --pred_len 720 --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
