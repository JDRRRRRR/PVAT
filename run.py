import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from dataset.data_factory import data_provider
from utils.tools import adjust_learning_rate
from utils.metrics import metric

import models.DLinear as DLinear
import models.PatchTST as PatchTST
import models.iTransformer as iTransformer
import models.TimeXer as TimeXer
import models.PVAT as PVAT
import models.TimesNet as TimesNet
import models.TimeMixer as TimeMixer

MODEL_REGISTRY = {
    'DLinear': DLinear,          
    'PatchTST': PatchTST,        
    'iTransformer': iTransformer, 
    'TimeXer': TimeXer,    
    'PVAT': PVAT,            
    'TimesNet': TimesNet,      
    'TimeMixer': TimeMixer,     
}

def get_device(gpu_id=None):

    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2) GPU support
        return torch.device('mps')
    return torch.device('cpu')

def build_model(args, enc_in):
    model_module = MODEL_REGISTRY.get(args.model)
    if model_module is None:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY.keys())}")

    if args.model == 'DLinear':
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=enc_in
        )
    elif args.model == 'PatchTST':
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers,
            enc_in=enc_in
        )
    elif args.model == 'iTransformer':
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers,
            enc_in=enc_in
        )
    elif args.model == 'TimeXer':
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            enc_in=enc_in,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers
        )
    elif args.model == 'PVAT':
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            n_vars=enc_in,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            en_d_ff=args.en_d_ff,
            de_d_ff=args.de_d_ff,
            en_layers=args.en_layers,
            de_layers=args.de_layers
        )
    elif args.model == 'TimesNet':
        from argparse import Namespace
        configs = Namespace(
            task_name='long_term_forecast',
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=enc_in,
            c_out=enc_in,
            d_model=args.d_model,
            d_ff=args.en_d_ff,
            dropout=args.dropout,
            e_layers=args.en_layers,
            top_k=args.top_k,
            num_kernels=args.num_kernels,
            embed='timeF',
            freq='h'
        )
        model = model_module.Model(configs)
    elif args.model == 'TimeMixer':
        from argparse import Namespace
        configs = Namespace(
            task_name='long_term_forecast',
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=enc_in,
            c_out=enc_in,
            d_model=args.d_model,
            d_ff=args.en_d_ff,
            dropout=args.dropout,
            e_layers=args.en_layers,
            down_sampling_layers=args.down_sampling_layers,
            down_sampling_window=args.down_sampling_window,
            down_sampling_method=args.down_sampling_method,
            channel_independence=args.channel_independence,
            decomp_method=args.decomp_method,
            moving_avg=args.moving_avg,
            top_k=args.top_k,
            use_norm=1,
            embed='timeF',
            freq='h'
        )
        model = model_module.Model(configs)

    return model.to(args.device)

def prepare_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, args):

    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)
    batch_x_mark = batch_x_mark.float().to(args.device)
    batch_y_mark = batch_y_mark.float().to(args.device)

    # Construct decoder input:
    # [label_len known values] + [pred_len zeros to be predicted]
    # This provides the decoder with context while masking future values
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

    return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp


def get_output_slice(outputs, batch_y, args):

    f_dim = -1 if args.features == 'MS' else 0

    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:]

    return outputs, batch_y


def train_epoch(model, train_loader, optimizer, loss_func, args):

    model.train()  
    train_losses = []

    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:

        optimizer.zero_grad()


        batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, args
        )

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs, batch_y = get_output_slice(outputs, batch_y, args)
        loss = loss_func(outputs, batch_y)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(train_losses)


def validate(model, val_loader, loss_func, args):

    model.eval()  
    val_losses = []

    with torch.no_grad():  
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, args
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y = get_output_slice(outputs, batch_y, args)

            loss = loss_func(outputs, batch_y)
            val_losses.append(loss.item())

    return np.mean(val_losses)


def test(model, test_loader, args):
    model.eval()
    preds, trues = [], []

    f_dim = -1 if args.features == 'MS' else 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, args
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]

            outputs = outputs.detach().cpu().numpy()[:, :, f_dim:]
            batch_y = batch_y.detach().cpu().numpy()[:, :, f_dim:]

            preds.append(outputs)
            trues.append(batch_y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    return preds, trues

def work_process(args):

    train_loader, enc_in = data_provider(args, flag='train')
    val_loader, _ = data_provider(args, flag='val')
    test_loader, _ = data_provider(args, flag='test')

    model = build_model(args, enc_in)
    loss_func = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.train_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_func, args)

        # Evaluate on validation set
        val_loss = validate(model, val_loader, loss_func, args)

        # Log progress
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.7f} Vali Loss: {val_loss:.7f}")

        # Adjust learning rate according to schedule
        adjust_learning_rate(optimizer, epoch + 1, args)

    preds, trues = test(model, test_loader, args)
    print(f'Test shape: {preds.shape}, {trues.shape}')

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print(f'Test shape: {preds.shape}, {trues.shape}')

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'MSE: {mse}, MAE: {mae}')

    os.makedirs(os.path.dirname(args.evaluation), exist_ok=True)

    with open(args.evaluation, 'a') as f:
        f.write(f'MSE: {mse}, MAE: {mae}\n')

def run():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--model', type=str, default='PAViT',
                        help=f'Model architecture. Options: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('--evaluation', type=str, default='./evaluation/',
                        help='Directory for saving evaluation results')
    parser.add_argument('--data', type=str, default='ETTh1',
                        help='Dataset name: [ETTh1, ETTm1, ETTh2, ETTm2, electricity, exchange_rate, traffic, weather]')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small/',
                        help='Root directory containing the data files')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID. If not specified, auto-detects best available device')

    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length (lookback window size)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Decoder start token length (overlap between encoder and decoder)')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch size for patch-based models. Must divide seq_len evenly')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction horizon length (how far ahead to forecast)')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task: M (multivariate), S (univariate), MS (multivariate-to-single)')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target variable name for S/MS forecasting tasks')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='Number of input variables (auto-detected from data)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('--factor', type=int, default=3,
                        help='Attention factor for ProbSparse attention')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--en_d_ff', type=int, default=2048,
                        help='Encoder feed-forward network dimension (typically 4x d_model)')
    parser.add_argument('--de_d_ff', type=int, default=2048,
                        help='Decoder feed-forward network dimension (typically 4x d_model)')
    parser.add_argument('--en_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--de_layers', type=int, default=2,
                        help='Number of decoder layers')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate for Adam optimizer')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='Loss function (currently only MSE supported)')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='Learning rate adjustment: type1 (exponential decay) or type2 (step schedule)')

    # TimesNet Specific Parameters
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top k frequencies for TimesNet/TimeMixer')
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='Number of kernels for Inception block in TimesNet')

    # TimeMixer Specific Parameters
    parser.add_argument('--down_sampling_layers', type=int, default=2,
                        help='Number of down sampling layers for TimeMixer')
    parser.add_argument('--down_sampling_window', type=int, default=2,
                        help='Down sampling window size for TimeMixer')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='Down sampling method: avg, max, conv')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='Channel independence for TimeMixer (0 or 1)')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='Decomposition method: moving_avg or dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='Moving average window size for decomposition')

    args = parser.parse_args()

    if args.seq_len % args.patch_len != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be divisible by patch_len ({args.patch_len})")

    args.device = get_device(args.gpu)
    print(f'Using device: {args.device}')

    file_name = (f"{args.model}{args.data}PatL{args.patch_len}PreL{args.pred_len}"
                 f"Fea{args.features}DM{args.d_model}ED{args.en_d_ff}"
                 f"DD{args.de_d_ff}EL{args.en_layers}DL{args.de_layers}")
    args.evaluation = os.path.join(args.evaluation, f'{file_name}.txt')

    print(args)

    work_process(args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fix_seed = 42

    torch.set_num_threads(4)

    random.seed(fix_seed)           # Python random
    torch.manual_seed(fix_seed)     # PyTorch CPU
    np.random.seed(fix_seed)        # NumPy

    run()
