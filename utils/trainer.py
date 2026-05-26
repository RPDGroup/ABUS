import os
import time
import json
import torch 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils.utils import *
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def compute_auc(scores: torch.Tensor, labels: torch.Tensor):
    """Compute the AUC score."""
    pred_probabilities = -scores.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    auc = metrics.roc_auc_score(label, pred_probabilities)
    return auc


def add_to_history(history: dict, train_loss: float, val_loss: float, train_auc: float, val_auc: float, save_model_dir: str):
    """Add the current epoch's metrics to the history."""
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))
    history['train_auc'].append(float(train_auc))
    history['val_auc'].append(float(val_auc))
    try:
        with open(os.path.join(save_model_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
    except Exception as e:
        log_info("History Saving Log", f"Writing history failed: {e}")
    return history


def plot_metrics(history: dict, save_model_dir: str, model_name: str):
    """Plot the training metrics."""
    plt.style.use('ggplot')
    figures_dir = os.path.join(save_model_dir, model_name+'_figs')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    # Loss
    plt.plot(history['train_loss'], marker='.', color='b', label='Train loss')
    plt.plot(history['val_loss'], marker='.', color='r', label='Validation loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(figures_dir, f'loss_{model_name}.png'))
    plt.close()
    # AUC
    plt.plot(history['train_auc'], marker='.', color='b', label='Train AUC')
    plt.plot(history['val_auc'], marker='.', color='r', label='Validation AUC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(figures_dir, f'auc_{model_name}.png'))
    

def val_one_epoch(model: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int, model_name: str, writer: SummaryWriter):
    """Validate the model for one epoch."""
    running_loss = 0.0
    running_auc = 0.0
    model.eval()
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            global_step = epoch * len(val_loader) + i
        
            image_tiles, labels, ppi = data
            image_tiles = np.reshape(image_tiles, (
                image_tiles.shape[0] * image_tiles.shape[1], 
                image_tiles.shape[2], 
                image_tiles.shape[3],
                image_tiles.shape[4]
            ))
            labels = np.reshape(labels, (labels.shape[0] * labels.shape[1]))
            image = image_tiles.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            
            scores, attn, loss = model(image, labels)
            auc = compute_auc(scores, labels)
            writer.add_scalar(f'{model_name}/val_batch_auc', auc, global_step)
            writer.add_scalar(f'{model_name}/val_batch_loss', loss.item(), global_step)
            running_loss += loss.item()
            running_auc += auc
            
    running_loss /= len(val_loader)
    running_auc /= len(val_loader)
    return running_loss, running_auc


def train_one_epoch(model: nn.Module, train_loader: DataLoader, device: torch.device, optimizer: Optimizer, epoch: int, model_name: str, writer: SummaryWriter):
    """Train the model for one epoch."""
    running_loss = 0.0
    running_auc = 0.0
    model.train()
    
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        global_step = epoch * len(train_loader) + i
        optimizer.zero_grad()
        
        image_tiles, labels, ppi = data
        image_tiles = np.reshape(image_tiles, (
            image_tiles.shape[0] * image_tiles.shape[1], 
            image_tiles.shape[2], 
            image_tiles.shape[3],
            image_tiles.shape[4]
        ))
        labels = np.reshape(labels, (labels.shape[0] * labels.shape[1]))
        image = image_tiles.to(device=device, dtype=torch.float)
        labels = labels.to(device)
        
        scores, attn, loss = model(image, labels)
        auc = compute_auc(scores, labels)
        running_auc += auc
        writer.add_scalar(f'{model_name}/train_batch_auc', auc, global_step)
        writer.add_scalar(f'{model_name}/train_batch_loss', loss.item(), global_step)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    running_loss /= len(train_loader)
    running_auc /= len(train_loader)
    return model, running_loss, running_auc
        
    



def fit(max_epochs: int, patience: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        optimizer: Optimizer, device: torch.device, model_name: str, save_model_dir: str, log_dir: str,
        save_model: bool = True):
    """Train the model with the given parameters."""
    writer = SummaryWriter(log_dir=log_dir) # log
    start = time.time()
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
        
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": []
    }
    
    model.to(device)
    
    min_loss = np.inf
    max_auc = 0
    not_improved = 0
    save_index = 0
    
    for epoch in range(max_epochs):
        log_info("Train Log", f"Epoch {epoch} / {max_epochs}")
        model, train_loss, train_auc = train_one_epoch(model, train_loader, device, optimizer, epoch, model_name, writer)
        val_loss, val_auc = val_one_epoch(model, val_loader, device, epoch, model_name, writer)
        writer.add_scalars(f'{model_name}/epoch_loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars(f'{model_name}/epoch_auc', {'train': train_auc, 'val': val_auc}, epoch)
        log_info("Train Log", f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_loss < min_loss:
            min_loss = val_loss
        if val_auc > max_auc:
            log_info("Train Log", f"epoch: {epoch} AUC increasing... {max_auc} >>> {val_auc}")
            max_auc = val_auc
            save_index = epoch
            not_improved = 0
        else:
            not_improved += 1
        
        if save_model:
            torch.save(model.state_dict(), os.path.join(save_model_dir, f"{model_name}_epoch_{epoch}.pth"))
        history = add_to_history(history, train_loss, val_loss, train_auc, val_auc, save_model_dir)
        
        if not_improved == patience:
            log_info("Train Log", f"Early Stopping {epoch} / {max_epochs}")
            break
    
    log_info("Train Log", "Training Finished")
    log_info("Train Log", f"Training cost {time.time() - start:.2f} s")
    plot_metrics(history, save_model_dir, model_name)
    log_info("Train Log", f"Best validation AUC: {max_auc:.4f} at epoch {save_index}")