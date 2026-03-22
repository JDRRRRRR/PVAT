import argparse
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset.data_factory import data_dict
from utils.tools import adjust_learning_rate
from utils.metrics import metric
import models.PVAT as PVAT


class VariableSubsetDataset(Dataset):

    def __init__(self, base_dataset, var_indices, target_idx=-1):
        self.base_dataset = base_dataset
        self.var_indices = var_indices
        self.target_idx = target_idx
        self.enc_in = len(var_indices) + 1  # auxiliary vars + target

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base_dataset[idx]

        # Select subset of variables (auxiliary + target)
        # Target is always the last column
        n_vars = seq_x.shape[-1]
        target_col = n_vars - 1 if self.target_idx == -1 else self.target_idx

        # Build selected columns: auxiliary subset + target
        selected_cols = list(self.var_indices) + [target_col]
        seq_x_subset = seq_x[:, selected_cols]
        seq_y_subset = seq_y[:, selected_cols]

        return seq_x_subset, seq_y_subset, seq_x_mark, seq_y_mark


def build_pvat_model(args, enc_in, use_ve_table=True):
    """Build PVAT model with or without VE Table."""
    model = PVAT.Model(
        seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
        n_vars=enc_in, d_model=args.d_model, dropout=args.dropout,
        factor=args.factor, n_heads=args.n_heads, en_d_ff=args.en_d_ff,
        de_d_ff=args.de_d_ff, en_layers=args.en_layers, de_layers=args.de_layers
    )

    if not use_ve_table:
        # Disable VE Table by zeroing out the embedding
        with torch.no_grad():
            model.auxiliary_encoder.variable_embedding_table.weight.zero_()
            # Freeze VE Table
            model.auxiliary_encoder.variable_embedding_table.weight.requires_grad = False

    return model


def get_pvat_shared_params(model, include_ve_table=True):
    """Get parameters that should be shared in federated learning."""
    shared_params = {}
    for name, param in model.named_parameters():
        # Local params: linear_patch, linear_var, forecast_head
        if 'linear_patch' in name or 'linear_var' in name or 'forecast_head' in name:
            continue
        # Optionally exclude VE Table
        if not include_ve_table and 'variable_embedding_table' in name:
            continue
        shared_params[name] = param.data.clone()
    return shared_params


def set_pvat_shared_params(model, shared_params):
    """Set shared parameters to model."""
    state_dict = model.state_dict()
    for name in shared_params:
        if name in state_dict:
            state_dict[name] = shared_params[name].clone()
    model.load_state_dict(state_dict)


def fedopt_aggregate(shared_params_list):
    """Average shared parameters from multiple nodes."""
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


def run_federated_ve_ablation(args, use_ve_table=True):
    
    print(f"\n=== Federated Training {'with' if use_ve_table else 'without'} VE Table ===")
    print(f"Dataset: {args.data}, Num Nodes: {args.num_nodes}, Aux Vars per Node: {args.num_aux_vars}")

    device = args.device
    Data = data_dict[args.data]

    # Load full dataset to get total number of variables
    full_train = Data(
        args=args, root_path=args.root_path, data_path=args.data + '.csv',
        flag='train', size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, timeenc=1, freq='h'
    )
    full_val = Data(
        args=args, root_path=args.root_path, data_path=args.data + '.csv',
        flag='val', size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, timeenc=1, freq='h'
    )
    full_test = Data(
        args=args, root_path=args.root_path, data_path=args.data + '.csv',
        flag='test', size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, timeenc=1, freq='h'
    )

    total_vars = full_train.enc_in
    num_aux_vars = total_vars - 1  # Exclude target
    vars_per_node = min(args.num_aux_vars, num_aux_vars)

    print(f"Total variables: {total_vars}, Auxiliary: {num_aux_vars}, Per node: {vars_per_node}")

    # Create nodes with different variable subsets
    nodes = []
    all_aux_indices = list(range(num_aux_vars))

    for node_id in range(args.num_nodes):
        # Randomly select auxiliary variables for this node
        node_var_indices = random.sample(all_aux_indices, vars_per_node)
        node_var_indices.sort()

        # Create subset datasets
        train_subset = VariableSubsetDataset(full_train, node_var_indices)
        val_subset = VariableSubsetDataset(full_val, node_var_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        # Build model for this node
        enc_in = train_subset.enc_in
        model = build_pvat_model(args, enc_in, use_ve_table=use_ve_table).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        nodes.append({
            'id': node_id,
            'var_indices': node_var_indices,
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'enc_in': enc_in
        })

        print(f"  Node {node_id}: vars {node_var_indices[:5]}... (total {len(node_var_indices)})")

    # Initialize global shared params from first node
    global_shared = get_pvat_shared_params(nodes[0]['model'], include_ve_table=use_ve_table)

    loss_func = nn.MSELoss()

    # Federated training loop
    for epoch in range(args.train_epochs):
        local_params_list = []

        for node in nodes:
            # Sync shared params
            set_pvat_shared_params(node['model'], global_shared)

            # Local training
            train_loss = train_epoch(
                node['model'], node['train_loader'],
                node['optimizer'], loss_func, args
            )

            # Collect shared params
            local_params = get_pvat_shared_params(node['model'], include_ve_table=use_ve_table)
            local_params_list.append(local_params)

        # Aggregate
        global_shared = fedopt_aggregate(local_params_list)

        # Validate (average across nodes)
        val_losses = []
        for node in nodes:
            set_pvat_shared_params(node['model'], global_shared)
            val_loss = validate(node['model'], node['val_loader'], loss_func, args)
            val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.6f}")

        # Adjust learning rate
        for node in nodes:
            adjust_learning_rate(node['optimizer'], epoch + 1, args)

    # Test on full dataset using first node's model (with all variables)
    # Re-create model with full variables for testing
    test_model = build_pvat_model(args, total_vars, use_ve_table=use_ve_table).to(device)
    set_pvat_shared_params(test_model, global_shared)

    test_loader = DataLoader(full_test, batch_size=args.batch_size, shuffle=False)
    preds, trues = test(test_model, test_loader, args)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f"Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}")

    return mse, mae


def run():
    parser = argparse.ArgumentParser(description='VE Table Ablation')
    parser.add_argument('--data', type=str, default='electricity',
                        choices=['electricity', 'traffic'])
    parser.add_argument('--root_path', type=str, default='dataset/electricity/')
    parser.add_argument('--evaluation', type=str, default='./evaluation/')
    parser.add_argument('--num_nodes', type=int, default=8)
    parser.add_argument('--num_aux_vars', type=int, default=50,
                        help='Number of auxiliary variables per node')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--en_d_ff', type=int, default=1024)
    parser.add_argument('--de_d_ff', type=int, default=1024)
    parser.add_argument('--en_layers', type=int, default=2)
    parser.add_argument('--de_layers', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lradj', type=str, default='type1')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set root_path based on data
    if args.data == 'electricity':
        args.root_path = 'dataset/electricity/'
    elif args.data == 'traffic':
        args.root_path = 'dataset/traffic/'

    print(f'Device: {args.device}')
    print(f'Dataset: {args.data}')

    results = {}

    # Test with different numbers of auxiliary variables
    aux_var_counts = [10, 30, 50, 100, 150]

    for num_aux in aux_var_counts:
        args.num_aux_vars = num_aux
        print(f"\n{'='*60}")
        print(f"Testing with {num_aux} auxiliary variables per node")
        print('='*60)

        # With VE Table
        mse_with, mae_with = run_federated_ve_ablation(args, use_ve_table=True)

        # Without VE Table
        mse_without, mae_without = run_federated_ve_ablation(args, use_ve_table=False)

        results[num_aux] = {
            'with_ve': {'mse': mse_with, 'mae': mae_with},
            'without_ve': {'mse': mse_without, 'mae': mae_without}
        }

        # Improvement
        imp_mse = (mse_without - mse_with) / mse_without * 100
        imp_mae = (mae_without - mae_with) / mae_without * 100
        print(f"\nImprovement with VE Table: MSE {imp_mse:.2f}%, MAE {imp_mae:.2f}%")

    # Summary
    print("\n" + "="*70)
    print("VE TABLE ABLATION SUMMARY")
    print("="*70)
    print(f"Dataset: {args.data}")
    print("-"*70)
    print(f"{'Aux Vars':<12} {'With VE (MSE/MAE)':<25} {'W/O VE (MSE/MAE)':<25} {'Improvement':<15}")
    print("-"*70)

    for num_aux in aux_var_counts:
        r = results[num_aux]
        with_str = f"{r['with_ve']['mse']:.4f}/{r['with_ve']['mae']:.4f}"
        without_str = f"{r['without_ve']['mse']:.4f}/{r['without_ve']['mae']:.4f}"
        imp = (r['without_ve']['mse'] - r['with_ve']['mse']) / r['without_ve']['mse'] * 100
        print(f"{num_aux:<12} {with_str:<25} {without_str:<25} {imp:>6.2f}%")

    # Save results
    os.makedirs(args.evaluation, exist_ok=True)
    with open(os.path.join(args.evaluation, f'ablation_ve_{args.data}.txt'), 'w') as f:
        f.write(f"VE Table Ablation - {args.data}\n")
        f.write(f"Nodes: {args.num_nodes}\n\n")
        f.write(f"{'Aux Vars':<12} {'With VE (MSE)':<15} {'W/O VE (MSE)':<15} {'Improvement':<15}\n")
        for num_aux in aux_var_counts:
            r = results[num_aux]
            imp = (r['without_ve']['mse'] - r['with_ve']['mse']) / r['without_ve']['mse'] * 100
            f.write(f"{num_aux:<12} {r['with_ve']['mse']:<15.6f} {r['without_ve']['mse']:<15.6f} {imp:>6.2f}%\n")


if __name__ == '__main__':
    fix_seed = 42
    torch.set_num_threads(4)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    run()
