import argparse
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.data_factory import data_provider, data_dict
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


def get_device(rank):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}')
    return torch.device('cpu')


def build_model(args, enc_in):
    model_module = MODEL_REGISTRY.get(args.model)
    if model_module is None:
        raise ValueError(f"Unknown model: {args.model}")

    if args.model == 'DLinear':
        model = model_module.Model(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in)
    elif args.model == 'PatchTST':
        model = model_module.Model(
            seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
            d_model=args.d_model, dropout=args.dropout, factor=args.factor,
            n_heads=args.n_heads, d_ff=args.en_d_ff, e_layers=args.en_layers, enc_in=enc_in
        )
    elif args.model == 'iTransformer':
        model = model_module.Model(
            seq_len=args.seq_len, pred_len=args.pred_len, d_model=args.d_model,
            dropout=args.dropout, factor=args.factor, n_heads=args.n_heads,
            d_ff=args.en_d_ff, e_layers=args.en_layers, enc_in=enc_in
        )
    elif args.model == 'TimeXer':
        model = model_module.Model(
            seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
            enc_in=enc_in, d_model=args.d_model, dropout=args.dropout,
            factor=args.factor, n_heads=args.n_heads, d_ff=args.en_d_ff, e_layers=args.en_layers
        )
    elif args.model == 'PVAT':
        model = model_module.Model(
            seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
            n_vars=enc_in, d_model=args.d_model, dropout=args.dropout,
            factor=args.factor, n_heads=args.n_heads, en_d_ff=args.en_d_ff,
            de_d_ff=args.de_d_ff, en_layers=args.en_layers, de_layers=args.de_layers
        )
    elif args.model == 'TimesNet':
        from argparse import Namespace
        configs = Namespace(
            task_name='long_term_forecast', seq_len=args.seq_len, label_len=args.label_len,
            pred_len=args.pred_len, enc_in=enc_in, c_out=enc_in, d_model=args.d_model,
            d_ff=args.en_d_ff, dropout=args.dropout, e_layers=args.en_layers,
            top_k=args.top_k, num_kernels=args.num_kernels, embed='timeF', freq='h'
        )
        model = model_module.Model(configs)
    elif args.model == 'TimeMixer':
        from argparse import Namespace
        configs = Namespace(
            task_name='long_term_forecast', seq_len=args.seq_len, label_len=args.label_len,
            pred_len=args.pred_len, enc_in=enc_in, c_out=enc_in, d_model=args.d_model,
            d_ff=args.en_d_ff, dropout=args.dropout, e_layers=args.en_layers,
            down_sampling_layers=args.down_sampling_layers, down_sampling_window=args.down_sampling_window,
            down_sampling_method=args.down_sampling_method, channel_independence=args.channel_independence,
            decomp_method=args.decomp_method, moving_avg=args.moving_avg, top_k=args.top_k,
            use_norm=1, embed='timeF', freq='h'
        )
        model = model_module.Model(configs)
    return model


def get_pvat_shared_params(model):
    """Get parameters that should be shared in federated learning for PVAT."""
    shared_params = {}
    for name, param in model.named_parameters():
        # Share: VE Table, Auxiliary Encoder, Target Decoder
        # Local: patch_embedding.linear_patch, auxiliary_encoder.linear_var, forecast_head
        if 'linear_patch' in name or 'linear_var' in name or 'forecast_head' in name:
            continue
        shared_params[name] = param
    return shared_params


def get_all_params(model):
    """Get all parameters for baseline models."""
    return {name: param for name, param in model.named_parameters()}


def fedopt_aggregate(global_params, local_params_list, server_lr=1.0):
    """FedOPT aggregation with server-side Adam."""
    with torch.no_grad():
        for name in global_params:
            # Average local updates
            stacked = torch.stack([lp[name] for lp in local_params_list])
            avg_param = stacked.mean(dim=0)
            # Server-side update (simplified FedOPT)
            global_params[name].copy_(avg_param)


def prepare_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, args):
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)
    batch_x_mark = batch_x_mark.float().to(args.device)
    batch_y_mark = batch_y_mark.float().to(args.device)
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
            outputs = outputs[:, -args.pred_len:, :].detach().cpu().numpy()[:, :, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()[:, :, f_dim:]
            preds.append(outputs)
            trues.append(batch_y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


class FederatedTrainer:
    """Simulates federated learning with multiple nodes."""

    def __init__(self, args, num_nodes=8):
        self.args = args
        self.num_nodes = num_nodes
        self.nodes = []
        self.global_model = None

    def setup_nodes(self):
        """Initialize models and data loaders for each node."""
        # Get full dataset info
        args = copy.deepcopy(self.args)
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Create dataset to get enc_in
        Data = data_dict[args.data]
        full_dataset = Data(
            args=args, root_path=args.root_path, data_path=args.data + '.csv',
            flag='train', size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features, target=args.target, timeenc=1, freq='h'
        )
        enc_in = full_dataset.enc_in

        # Build global model
        self.global_model = build_model(args, enc_in).to(args.device)

        # Create nodes with distributed samplers
        for node_id in range(self.num_nodes):
            node_args = copy.deepcopy(args)
            node_args.device = args.device

            # Create dataset with subset for this node
            train_dataset = Data(
                args=node_args, root_path=node_args.root_path,
                data_path=node_args.data + '.csv', flag='train',
                size=[node_args.seq_len, node_args.label_len, node_args.pred_len],
                features=node_args.features, target=node_args.target, timeenc=1, freq='h'
            )

            # Simulate DistributedSampler by splitting indices
            total_samples = len(train_dataset)
            samples_per_node = total_samples // self.num_nodes
            start_idx = node_id * samples_per_node
            end_idx = start_idx + samples_per_node if node_id < self.num_nodes - 1 else total_samples

            indices = list(range(start_idx, end_idx))
            subset = torch.utils.data.Subset(train_dataset, indices)

            train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=False)

            # Create local model (copy of global)
            local_model = build_model(node_args, enc_in).to(node_args.device)
            local_model.load_state_dict(self.global_model.state_dict())

            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)

            self.nodes.append({
                'id': node_id,
                'model': local_model,
                'train_loader': train_loader,
                'optimizer': optimizer,
                'args': node_args
            })

        # Create validation and test loaders
        val_dataset = Data(
            args=args, root_path=args.root_path, data_path=args.data + '.csv',
            flag='val', size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features, target=args.target, timeenc=1, freq='h'
        )
        test_dataset = Data(
            args=args, root_path=args.root_path, data_path=args.data + '.csv',
            flag='test', size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features, target=args.target, timeenc=1, freq='h'
        )

        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"Initialized {self.num_nodes} federated nodes")
        print(f"Samples per node: ~{len(train_dataset) // self.num_nodes}")

    def train_round(self, loss_func):
        """Execute one round of federated training."""
        local_params_list = []

        # Local training on each node
        for node in self.nodes:
            # Sync with global model
            if self.args.model == 'PVAT':
                # Only sync shared parameters
                global_shared = get_pvat_shared_params(self.global_model)
                local_state = node['model'].state_dict()
                for name in global_shared:
                    local_state[name] = global_shared[name].clone()
                node['model'].load_state_dict(local_state)
            else:
                # Sync all parameters for baselines
                node['model'].load_state_dict(self.global_model.state_dict())

            # Local training
            train_loss = train_epoch(
                node['model'], node['train_loader'],
                node['optimizer'], loss_func, node['args']
            )

            # Collect parameters for aggregation
            if self.args.model == 'PVAT':
                local_params_list.append(get_pvat_shared_params(node['model']))
            else:
                local_params_list.append(get_all_params(node['model']))

        # Server aggregation (FedOPT)
        if self.args.model == 'PVAT':
            global_shared = get_pvat_shared_params(self.global_model)
            fedopt_aggregate(global_shared, local_params_list)
        else:
            global_params = get_all_params(self.global_model)
            fedopt_aggregate(global_params, local_params_list)

        return train_loss

    def train(self):
        """Full federated training loop."""
        loss_func = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            train_loss = self.train_round(loss_func)
            val_loss = validate(self.global_model, self.val_loader, loss_func, self.args)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")

            # Adjust learning rate for all nodes
            for node in self.nodes:
                adjust_learning_rate(node['optimizer'], epoch + 1, self.args)

        # Test
        preds, trues = test(self.global_model, self.test_loader, self.args)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'Test Results - MSE: {mse:.6f}, MAE: {mae:.6f}')

        return mse, mae


def run():
    parser = argparse.ArgumentParser(description='Federated Time Series Forecasting')
    parser.add_argument('--model', type=str, default='PVAT')
    parser.add_argument('--evaluation', type=str, default='./evaluation/')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small/')
    parser.add_argument('--num_nodes', type=int, default=8)

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--en_d_ff', type=int, default=2048)
    parser.add_argument('--de_d_ff', type=int, default=2048)
    parser.add_argument('--en_layers', type=int, default=2)
    parser.add_argument('--de_layers', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lradj', type=str, default='type1')

    # TimesNet/TimeMixer params
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--down_sampling_layers', type=int, default=2)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--moving_avg', type=int, default=25)

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {args.device}')
    print(f'Model: {args.model}, Data: {args.data}, Nodes: {args.num_nodes}')
    print(args)

    # Run federated training
    trainer = FederatedTrainer(args, num_nodes=args.num_nodes)
    trainer.setup_nodes()
    mse, mae = trainer.train()

    # Save results
    os.makedirs(args.evaluation, exist_ok=True)
    file_name = f"Fed_{args.model}_{args.data}_N{args.num_nodes}_P{args.pred_len}.txt"
    with open(os.path.join(args.evaluation, file_name), 'a') as f:
        f.write(f'MSE: {mse}, MAE: {mae}\n')


if __name__ == '__main__':
    fix_seed = 42
    torch.set_num_threads(4)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    run()
