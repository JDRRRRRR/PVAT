#!/bin/bash
# TimeMixer Model Experiments
# Multi-scale Season/Trend Mixing for Time-Series Forecasting

# ETTh1 Dataset
python run.py --model TimeMixer --data ETTh1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTh1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTh1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTh1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# ETTh2 Dataset
python run.py --model TimeMixer --data ETTh2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTh2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTh2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTh2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# ETTm1 Dataset
python run.py --model TimeMixer --data ETTm1 --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTm1 --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTm1 --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTm1 --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# ETTm2 Dataset
python run.py --model TimeMixer --data ETTm2 --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTm2 --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTm2 --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data ETTm2 --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# Electricity Dataset
python run.py --model TimeMixer --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data electricity --root_path ./dataset/electricity/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# Exchange Rate Dataset
python run.py --model TimeMixer --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data exchange_rate --root_path ./dataset/exchange_rate/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# Traffic Dataset
python run.py --model TimeMixer --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data traffic --root_path ./dataset/traffic/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1

# Weather Dataset
python run.py --model TimeMixer --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 96 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 192 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 336 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
python run.py --model TimeMixer --data weather --root_path ./dataset/weather/ --features M --target OT --seq_len 96 --pred_len 720 --d_model 32 --en_d_ff 32 --en_layers 2 --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1
