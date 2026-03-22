#!/bin/bash
# TimesNet Model Experiments
# 2D Convolution based Time-Series Forecasting with FFT Period Detection

# ETTh1 Dataset
python run.py --model TimesNet --data ETTh1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTh1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTh1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTh1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# ETTh2 Dataset
python run.py --model TimesNet --data ETTh2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTh2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTh2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTh2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# ETTm1 Dataset
python run.py --model TimesNet --data ETTm1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTm1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTm1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTm1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# ETTm2 Dataset
python run.py --model TimesNet --data ETTm2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTm2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTm2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data ETTm2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# Electricity Dataset
python run.py --model TimesNet --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# Exchange Rate Dataset
python run.py --model TimesNet --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# Traffic Dataset
python run.py --model TimesNet --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6

# Weather Dataset
python run.py --model TimesNet --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
python run.py --model TimesNet --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 64 --en_d_ff 64 --en_layers 2 --top_k 5 --num_kernels 6
