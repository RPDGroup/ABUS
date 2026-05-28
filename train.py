import os
import sys
import torch
import random
import argparse
import numpy as np
from utils.utils import *
from model.ABUS import ABUS
from utils.trainer import fit
from utils.dataset import AbAg_Dataset
from torch.utils.data import DataLoader
from model.ViT_pytorch import get_ml_config


WORK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORK_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="ABUS Training")
    parser.add_argument('--model_name', type=str, default='ABUS', help='Model name')
    parser.add_argument('--data_dir', type=str, default=f'{WORK_DIR}/data/AbAg_PDB2025', help='Dataset directory')
    parser.add_argument('--train_list', type=str, default=None, help='Train csv file')
    parser.add_argument('--val_list', type=str, default=None, help='Validation csv file')
    return parser.parse_args()


ABUS_CONFIG = {
    # Model Configuration    
    # Common
    'hidden_size': 16,
    'img_size': 32,
    'patch_size': 4,
    
    # Swin
    'window_size': 4,
    'swin_depths': [2, 2],
    'swin_num_heads': [4, 8],
    'drop_path_rate': 0.1,
    'shift_size': 2,
    'use_relative_position_bias': True,
    
    # Encoder   
    'dim_head': 16,
    'dropout': 0,
    'attn_dropout': 0,
    'n_heads': 8,
    'transformer_depth': 8,
    
    # Train Configuration
    'max_epochs': 200,
    'lr': 0.0001,
    'batch_size': 1,
    'patience': 10,
    'seed': 7272,
    'temperature': 0.5,
    'margin': 0,
    'weight_decay': 0.0001,
    'n_features': 13,
    'features_subset': list(range(13)),
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def train(search_space: dict, train_list_file: str, val_list_file: str, model_dir: str, model_name: str, data_dir: str, log_dir: str, config: dict) -> None:
    log_info("Data Loading", "Computing mean and std of data...")
    train_list, val_list, mean, std = compute_std_mean(train_list_file, val_list_file, config, model_dir, search_space)
    
    # Dataset
    log_info("Dataset Log", f"Building train dataset...")
    train_db = AbAg_Dataset(ppi_list=train_list, is_train=True, data_dir=data_dir, std=std, mean=mean, features_subset=search_space['features_subset'])
    log_info("Dataset Log", f"Building val dataset...")
    val_db = AbAg_Dataset(ppi_list=val_list, is_train=False, data_dir=data_dir, std=std, mean=mean, features_subset=search_space['features_subset'])
    
    # Dataloader
    train_loader = DataLoader(train_db, batch_size=search_space['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_db, batch_size=search_space['batch_size'], shuffle=False, pin_memory=True)

    # Model
    model_config = get_ml_config(search_space)
    device = search_space['device']
    model = ABUS(model_config, img_size=search_space['img_size'], margin=search_space['margin'], temperature=search_space['temperature']).float()
    model = model.to(device)
    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    log_info("Model Log", f"Load model {model_name} with {n_params} parameters.")
    save_model_config(model_dir, model_name, n_params, search_space)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=search_space['lr'], weight_decay=search_space['weight_decay'])

    # training and validation
    log_info("Train Log", f"Using device {device} for training and validation.")
    fit(
        max_epochs=search_space['max_epochs'],
        patience=search_space['patience'],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        save_model_dir=model_dir,
        log_dir=log_dir,
        save_model=True
    )


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    data_dir = args.data_dir
    model_dir = f'{WORK_DIR}/save_model/{model_name}'
    log_dir = f'{WORK_DIR}/train_log/{model_name}'
    train_list_file = args.train_list or f'{args.data_dir}/train.csv'
    val_list_file = args.val_list or f'{args.data_dir}/val.csv'
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    set_seed(ABUS_CONFIG['seed'])
    config = make_config(data_dir)
    train(
        search_space = ABUS_CONFIG,
        train_list_file = train_list_file,
        val_list_file = val_list_file,
        model_dir = model_dir,
        model_name = model_name,
        data_dir = data_dir,
        log_dir = log_dir,
        config = config
    )