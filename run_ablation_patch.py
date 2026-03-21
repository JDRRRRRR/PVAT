import argparse
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.data_factory import data_dict
from utils.tools import adjust_learning_rate
from utils.metrics import metric
import models.PVAT as PVAT


def build_pvat_model(args, enc_in):
    return PVAT.Model(
        seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
        n_vars=enc_in, d_model=args.d_model, dropout=args.dropout,
        factor=args.factor, n_heads=args.n_heads, en_d_ff=args.en_d_ff,
        de_d_ff=args.de_d_ff, en_layers=args.en_layers, de_layers=args.de_layers
    )


def get_pvat_shared_params(model):
    shared_params = {}
    for name, param in model.named_parameters():
        if 'linear_patch' in name or 'linear_var' in name or 'forecast_head' in name:
            continue
        shared_params[name] = param.data.clone()
    return shared_params


def set_pvat_shared_params(model, shared_params):
    state_dict = model.state_dict()
    for name in shared_params:
        state_dict[name] = shared_params[name].clone()
    model.load_state_dict(state_dict)


def fedopt_aggregate(shared_params_list):
    aggregated = {}
    for name in shared_params_list[0]:
        stacked = torch.stack([sp[name] for sp in shared_params_list])
        aggregated[name] = stacked.mean(dim=0)
    return aggregated


def prepare_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, args):
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)
    batch_x_mark = batch_x_mark.float().to(args.device)
    batch_y_mark = batch_y_mark.float().to(args.device)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
    return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp


def train_epoch(model, train_loader, optimizer, loss_func, args):
    model.train()
    train_losses = []
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        optimizer.zero_grad()
        batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, args
        )
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:]
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
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
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
            outputs = outputs[:, -args.pred_len:, :].detach().cpu().numpy()[:, :, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()[:, :, f_dim:]
            preds.append(outputs)
            trues.append(batch_y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


def train_single_granularity(args, data_name):
    """Train on single granularity (ETTh or ETTm only)."""
    print(f"\n=== Training on {data_name} only ===")

    node_args = copy.deepcopy(args)
    node_args.data = data_name

    # Adjust root_path based on data
    node_args.root_path = 'dataset/ETT-small/'

    # For hourly data, adjust pred_len to match physical time
    # ETTm: pred_len 96 = 24 hours, ETTh: pred_len 24 = 24 hours
    if 'h' in data_name:
        # Hourly: divide pred_len by 4 to match physical time
        node_args.pred_len = args.pred_len // 4
        node_args.seq_len = args.seq_len // 4 if args.seq_len >= 96 else args.seq_len
        node_args.label_len = args.label_len // 4 if args.label_len >= 48 else args.label_len
        node_args.patch_len = max(4, args.patch_len // 4)

    Data = data_dict[data_name]

    train_dataset = Data(
        args=node_args, root_path=node_args.root_path,
        data_path=data_name + '.csv', flag='train',
        size=[node_args.seq_len, node_args.label_len, node_args.pred_len],
        features=node_args.features, target=node_args.target, timeenc=1, freq='h'
    )
    val_dataset = Data(
        args=node_args, root_path=node_args.root_path,
        data_path=data_name + '.csv', flag='val',
        size=[node_args.seq_len, node_args.label_len, node_args.pred_len],
        features=node_args.features, target=node_args.target, timeenc=1, freq='h'
    )
    test_dataset = Data(
        args=node_args, root_path=node_args.root_path,
        data_path=data_name + '.csv', flag='test',
        size=[node_args.seq_len, node_args.label_len, node_args.pred_len],
        features=node_args.features, target=node_args.target, timeenc=1, freq='h'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    enc_in = train_dataset.enc_in
    model = build_pvat_model(node_args, enc_in).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.MSELoss()

    for epoch in range(args.train_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_func, node_args)
        val_loss = validate(model, val_loader, loss_func, node_args)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
        adjust_learning_rate(optimizer, epoch + 1, args)

    preds, trues = test(model, test_loader, node_args)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    mae, mse, rmse, mape, mspe = metric(preds, trues)

    print(f"{data_name} Results - MSE: {mse:.6f}, MAE: {mae:.6f}")
    return mse, mae


def train_cross_granularity_federated(args, data_h, data_m):
    """
    Federated training across different sampling rates.
    ETTh (1-hour) + ETTm (15-minute) with aligned physical time intervals.
    """
    print(f"\n=== Federated Training: {data_h} + {data_m} ===")

    device = args.device

    # Setup hourly node (ETTh)
    args_h = copy.deepcopy(args)
    args_h.data = data_h
    args_h.root_path = 'dataset/ETT-small/'
    # Adjust for hourly: same physical time = 1/4 samples
    args_h.pred_len = args.pred_len // 4
    args_h.seq_len = args.seq_len // 4 if args.seq_len >= 96 else 24
    args_h.label_len = args.label_len // 4 if args.label_len >= 48 else 12
    args_h.patch_len = max(4, args.patch_len // 4)

    # Setup minute node (ETTm)
    args_m = copy.deepcopy(args)
    args_m.data = data_m
    args_m.root_path = 'dataset/ETT-small/'
    # Keep original settings for 15-minute data

    Data_h = data_dict[data_h]
    Data_m = data_dict[data_m]

    # Create datasets
    train_h = Data_h(args=args_h, root_path=args_h.root_path, data_path=data_h+'.csv',
                     flag='train', size=[args_h.seq_len, args_h.label_len, args_h.pred_len],
                     features=args_h.features, target=args_h.target, timeenc=1, freq='h')
    train_m = Data_m(args=args_m, root_path=args_m.root_path, data_path=data_m+'.csv',
                     flag='train', size=[args_m.seq_len, args_m.label_len, args_m.pred_len],
                     features=args_m.features, target=args_m.target, timeenc=1, freq='h')

    val_h = Data_h(args=args_h, root_path=args_h.root_path, data_path=data_h+'.csv',
                   flag='val', size=[args_h.seq_len, args_h.label_len, args_h.pred_len],
                   features=args_h.features, target=args_h.target, timeenc=1, freq='h')
    val_m = Data_m(args=args_m, root_path=args_m.root_path, data_path=data_m+'.csv',
                   flag='val', size=[args_m.seq_len, args_m.label_len, args_m.pred_len],
                   features=args_m.features, target=args_m.target, timeenc=1, freq='h')

    test_h = Data_h(args=args_h, root_path=args_h.root_path, data_path=data_h+'.csv',
                    flag='test', size=[args_h.seq_len, args_h.label_len, args_h.pred_len],
                    features=args_h.features, target=args_h.target, timeenc=1, freq='h')
    test_m = Data_m(args=args_m, root_path=args_m.root_path, data_path=data_m+'.csv',
                    flag='test', size=[args_m.seq_len, args_m.label_len, args_m.pred_len],
                    features=args_m.features, target=args_m.target, timeenc=1, freq='h')

    loader_h_train = DataLoader(train_h, batch_size=args.batch_size, shuffle=True)
    loader_m_train = DataLoader(train_m, batch_size=args.batch_size, shuffle=True)
    loader_h_val = DataLoader(val_h, batch_size=args.batch_size, shuffle=False)
    loader_m_val = DataLoader(val_m, batch_size=args.batch_size, shuffle=False)
    loader_h_test = DataLoader(test_h, batch_size=args.batch_size, shuffle=False)
    loader_m_test = DataLoader(test_m, batch_size=args.batch_size, shuffle=False)

    enc_in = train_h.enc_in  # Same for both ETT datasets

    # Build models for each node (different patch_len due to different sampling rates)
    model_h = build_pvat_model(args_h, enc_in).to(device)
    model_m = build_pvat_model(args_m, enc_in).to(device)

    optimizer_h = torch.optim.Adam(model_h.parameters(), lr=args.learning_rate)
    optimizer_m = torch.optim.Adam(model_m.parameters(), lr=args.learning_rate)
    loss_func = nn.MSELoss()

    # Initialize shared params from model_m
    global_shared = get_pvat_shared_params(model_m)

    for epoch in range(args.train_epochs):
        # Sync shared params to both models
        set_pvat_shared_params(model_h, global_shared)
        set_pvat_shared_params(model_m, global_shared)

        # Local training on each node
        train_loss_h = train_epoch(model_h, loader_h_train, optimizer_h, loss_func, args_h)
        train_loss_m = train_epoch(model_m, loader_m_train, optimizer_m, loss_func, args_m)

        # Collect shared params
        shared_h = get_pvat_shared_params(model_h)
        shared_m = get_pvat_shared_params(model_m)

        # Aggregate
        global_shared = fedopt_aggregate([shared_h, shared_m])

        # Validate on both
        set_pvat_shared_params(model_h, global_shared)
        set_pvat_shared_params(model_m, global_shared)
        val_loss_h = validate(model_h, loader_h_val, loss_func, args_h)
        val_loss_m = validate(model_m, loader_m_val, loss_func, args_m)

        print(f"Epoch {epoch+1}: Train(h/m): {train_loss_h:.5f}/{train_loss_m:.5f}, "
              f"Val(h/m): {val_loss_h:.5f}/{val_loss_m:.5f}")

        adjust_learning_rate(optimizer_h, epoch + 1, args)
        adjust_learning_rate(optimizer_m, epoch + 1, args)

    # Test on both
    set_pvat_shared_params(model_h, global_shared)
    set_pvat_shared_params(model_m, global_shared)

    preds_h, trues_h = test(model_h, loader_h_test, args_h)
    preds_m, trues_m = test(model_m, loader_m_test, args_m)

    preds_h = preds_h.reshape(-1, preds_h.shape[-2], preds_h.shape[-1])
    trues_h = trues_h.reshape(-1, trues_h.shape[-2], trues_h.shape[-1])
    preds_m = preds_m.reshape(-1, preds_m.shape[-2], preds_m.shape[-1])
    trues_m = trues_m.reshape(-1, trues_m.shape[-2], trues_m.shape[-1])

    mae_h, mse_h, _, _, _ = metric(preds_h, trues_h)
    mae_m, mse_m, _, _, _ = metric(preds_m, trues_m)

    # Average metrics
    mse_avg = (mse_h + mse_m) / 2
    mae_avg = (mae_h + mae_m) / 2

    print(f"\nFederated Results:")
    print(f"  {data_h}: MSE={mse_h:.6f}, MAE={mae_h:.6f}")
    print(f"  {data_m}: MSE={mse_m:.6f}, MAE={mae_m:.6f}")
    print(f"  Average: MSE={mse_avg:.6f}, MAE={mae_avg:.6f}")

    return mse_avg, mae_avg, mse_h, mae_h, mse_m, mae_m


def run():
    parser = argparse.ArgumentParser(description='Patch Embedding Ablation')
    parser.add_argument('--ett_version', type=str, default='1', choices=['1', '2'])
    parser.add_argument('--evaluation', type=str, default='./evaluation/')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--en_d_ff', type=int, default=512)
    parser.add_argument('--de_d_ff', type=int, default=512)
    parser.add_argument('--en_layers', type=int, default=1)
    parser.add_argument('--de_layers', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lradj', type=str, default='type1')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {args.device}')
    print(f'ETT Version: {args.ett_version}')

    # Dataset names
    data_h = f'ETTh{args.ett_version}'  # 1-hour sampling
    data_m = f'ETTm{args.ett_version}'  # 15-minute sampling

    results = {}

    # 1. Train on hourly data only
    mse_h, mae_h = train_single_granularity(args, data_h)
    results['hourly_only'] = {'mse': mse_h, 'mae': mae_h}

    # 2. Train on minute data only
    mse_m, mae_m = train_single_granularity(args, data_m)
    results['minute_only'] = {'mse': mse_m, 'mae': mae_m}

    # 3. Federated training (cross-granularity)
    mse_fed, mae_fed, mse_fed_h, mae_fed_h, mse_fed_m, mae_fed_m = \
        train_cross_granularity_federated(args, data_h, data_m)
    results['federated'] = {
        'mse': mse_fed, 'mae': mae_fed,
        'mse_h': mse_fed_h, 'mae_h': mae_fed_h,
        'mse_m': mse_fed_m, 'mae_m': mae_fed_m
    }

    # Summary
    print("\n" + "="*60)
    print("PATCH EMBEDDING ABLATION SUMMARY")
    print("="*60)
    print(f"ETT{args.ett_version} Dataset (pred_len={args.pred_len})")
    print("-"*60)
    print(f"Hourly Only:   MSE={results['hourly_only']['mse']:.6f}, MAE={results['hourly_only']['mae']:.6f}")
    print(f"Minute Only:   MSE={results['minute_only']['mse']:.6f}, MAE={results['minute_only']['mae']:.6f}")
    print(f"Federated:     MSE={results['federated']['mse']:.6f}, MAE={results['federated']['mae']:.6f}")
    print("-"*60)

    # Improvement calculation
    imp_vs_h_mse = (results['hourly_only']['mse'] - results['federated']['mse']) / results['hourly_only']['mse'] * 100
    imp_vs_m_mse = (results['minute_only']['mse'] - results['federated']['mse']) / results['minute_only']['mse'] * 100
    print(f"Improvement vs Hourly: {imp_vs_h_mse:.2f}% MSE reduction")
    print(f"Improvement vs Minute: {imp_vs_m_mse:.2f}% MSE reduction")

    # Save results
    os.makedirs(args.evaluation, exist_ok=True)
    with open(os.path.join(args.evaluation, f'ablation_patch_ETT{args.ett_version}.txt'), 'w') as f:
        f.write(f"Patch Embedding Ablation - ETT{args.ett_version}\n")
        f.write(f"pred_len={args.pred_len}\n\n")
        f.write(f"Hourly Only: MSE={results['hourly_only']['mse']:.6f}, MAE={results['hourly_only']['mae']:.6f}\n")
        f.write(f"Minute Only: MSE={results['minute_only']['mse']:.6f}, MAE={results['minute_only']['mae']:.6f}\n")
        f.write(f"Federated:   MSE={results['federated']['mse']:.6f}, MAE={results['federated']['mae']:.6f}\n")
        f.write(f"\nImprovement vs Hourly: {imp_vs_h_mse:.2f}%\n")
        f.write(f"Improvement vs Minute: {imp_vs_m_mse:.2f}%\n")


if __name__ == '__main__':
    fix_seed = 42
    torch.set_num_threads(4)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    run()
