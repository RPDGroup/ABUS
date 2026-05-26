import os
import re
import random
import plotly
import numpy as np
import pandas as pd
from scipy import ndimage
from utils.utils import *
from plotly import graph_objs as go
from torch.utils.data import Dataset
from multiprocessing import cpu_count
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor, as_completed


class AbAg_Dataset(Dataset):
    """Antibody and Antigen Dataset"""
    def __init__(self, ppi_list: list, is_train: bool, data_dir: str, std: np.ndarray, mean: np.ndarray, features_subset: list):
        self.ppi_list = ppi_list
        self.is_train = is_train
        self.data_dir = data_dir 
        self.static_grid_dir = os.path.join(data_dir, "static")
        self.docked_grid_dir = os.path.join(data_dir, "docked")
        self.features_subset = features_subset
        self.std = std
        self.mean = mean
        self.pos_dict, self.neg_dict = self._select_pos_neg_model(ppi_list)
        
        count_pos = sum([len(self.pos_dict[ppi]) for ppi in self.pos_dict.keys()]) + len(self.pos_dict.keys())
        count_neg = sum([len(self.neg_dict[ppi]) for ppi in self.neg_dict.keys()])
        
        log_info("Dataset Log", f"{'Train' if self.is_train else 'Validation'} Dataset build successfully. Total PPI: {len(self.ppi_list)}, Total Pos: {count_pos}, Total Neg: {count_neg}")
        if self.features_subset is not None:
            log_info("Dataset Log", f"We will using feature subset: {self.features_subset}")
        
        ppi = random.choice(self.ppi_list)
        random_ppi = os.path.join(self.static_grid_dir, ppi, f"{ppi}.npy")
        random_image = np.load(random_ppi, allow_pickle=True)
        self.background_mask = get_background_mask(random_image)
        self.n_features = random_image.shape[-1]
    
    def __len__(self):
        """Get the total number of PPI"""
        return len(self.ppi_list)
    
    def _select_pos_neg_model(self, ppi_list):
        """Select positive and negative models for each PPI"""
        pos_dict, neg_dict, ppi_list_updated = {}, {}, []
        log_info("Dataset Log", f"Start load positive and negative models for each PPI.")
                    
        tasks = [(ppi, self.data_dir, self.docked_grid_dir) for ppi in ppi_list]
        with ProcessPoolExecutor(max_workers = min(len(tasks), min(64, cpu_count()))) as exe:
            futures = {
                exe.submit(select_one_ppi, task): task[0] for task in tasks
            }
            for future in as_completed(futures):
                try:
                    ppi = futures[future]
                    ppi_result, pos_models, neg_models = future.result()
                except Exception as e:
                    log_info("Warn", f"Task failed: {str(e)}", Colour.RED)
                    continue
                if len(neg_models) > 1:
                    pos_dict[ppi_result] = pos_models
                    neg_dict[ppi_result] = neg_models
                    ppi_list_updated.append(ppi_result)
        
        self.ppi_list = ppi_list_updated
        return pos_dict, neg_dict
    
    def _rotate(self, grid: np.ndarray):
        """Rotate the grid"""
        angle = np.random.randint(low=1, high=360)
        for feature in range(0, grid.shape[-1]):
            grid[:, :, feature] = ndimage.rotate(grid[:, :, feature], angle, reshape=False)
        return grid
    
    def __getitem__(self, i):
        """Get the i-th item"""
        ppi = self.ppi_list[i]
        pos_models, neg_models = self.pos_dict[ppi], self.neg_dict[ppi]
        if self.is_train:
            random.shuffle(pos_models)
            random.shuffle(neg_models)
        pos_grids, neg_grids = [], []

        pos_paths = []
        neg_paths = []
        pos_paths = [os.path.join(self.docked_grid_dir, "07-grid", ppi, model_i, f"{model_i}.npy") for model_i in pos_models]
        neg_paths = [os.path.join(self.docked_grid_dir, "07-grid", ppi, model_i, f"{model_i}.npy") for model_i in neg_models]
        
        static_npy_file = os.path.join(self.static_grid_dir, ppi, f"{ppi}.npy")
        static_grid1 = self._rotate(np.load(static_npy_file, allow_pickle=True))
        static_grid2 = self._rotate(np.load(static_npy_file, allow_pickle=True))
        
        pos_grids = [self._rotate(np.load(pos_path, allow_pickle=True)) for pos_path in pos_paths if os.path.exists(pos_path)] + [static_grid1, static_grid2]
        neg_grids = [self._rotate(np.load(neg_path, allow_pickle=True)) for neg_path in neg_paths if os.path.exists(neg_path)]
        
        labels = np.array([1] * len(pos_grids) + [0] * len(neg_grids))
        grid = pos_grids + neg_grids

        # Standardize the grid
        grid = np.swapaxes(grid, -1, 1).astype(np.float32)
        if self.features_subset:
            grid = grid[:, self.features_subset, :, :]
        if self.mean is not None and self.std is not None:
            for feature in range(grid.shape[1]):
                grid[:, feature, :, :] = (grid[:, feature, :, :] - self.mean[feature]) / self.std[feature]
            grid = np.logical_and(grid, self.background_mask) * grid
            
        return grid, labels, ppi
        
                
class ABUS_Dataset(Dataset):
    """ABUS Dataset"""
    def __init__(self, grid_dir, ppi_list, attn=None):
        # Empirically learned mean and standard deviations:
        mean_array = [
            0.052493957488, 0.040719406235, -0.031886160740, -0.020656846348, 
            -0.220359811231, 0.052493957488, 0.040719406235, -0.031886160740, 
            -0.020656846348, -0.220359811231, 11.018893716025, 0.169624719337, 
            0.169624719337
        ]
        std_array = [
            0.440274202052, 0.137494676127, 0.189911097296, 0.214182060165, 
            0.520825546689, 0.440274202052, 0.137494676127, 0.189911097296, 
            0.214182060165, 0.520825546689, 8.105094235590, 0.178054469307, 
            0.178054469307
        ]
        
        all_grids = []
        ppi_to_idx = {}
        
        i = 0
        for ppi in ppi_list:
            if os.path.exists(os.path.join(grid_dir, f"{ppi}.npy")):
                ppi_to_idx[ppi] = i
                i += 1
                
                grid = np.load(os.path.join(grid_dir, f"{ppi}.npy"), allow_pickle=True)
                all_grids.append(grid)
                
        self.ppi_to_idx = ppi_to_idx
        
        background_mask = get_background_mask(grid)
        
        grid = np.stack(all_grids, axis=0)
        grid = np.swapaxes(grid, -1, 1).astype(np.float32)
        
        log_info("Dataset Log", f"Total shape of grids: {grid.shape}")
        
        # Interactino maps:
        for feature_i in range(grid.shape[1]):
            grid[:, feature_i, :, :] = (grid[:, feature_i, :, :] - mean_array[feature_i]) / std_array[feature_i]
            # Mask out values that are out of the radius:
            grid = np.logical_and(grid, background_mask) * grid
            
        self.grid = grid
        self.grid_dir = grid_dir
        self.ppi_list = ppi_list
        
    def vis_patch(self, ppi, html_path=None, attn=None):
        feature_pairs = {
            'shape_index': (0, 5),
            'ddc': (1, 6),
            'electrostatics': (2, 7),
            'charge': (3, 8),
            'hydrophobicity': (4, 9),
            'RASA': (11, 12),
            'patch_dist': (10,),
        }
        grid_dir = self.grid_dir
        resnames_path = os.path.join(grid_dir, ppi, f"{ppi}_resnames.npy")
        patch_path = os.path.join(grid_dir, ppi, f"{ppi}.npy")
        patch_np = np.load(patch_path, allow_pickle=True)

        patch_resnames = np.load(resnames_path, allow_pickle=True)
        n_feat = int(patch_np.shape[-1] / 2)
        key_names = list(feature_pairs.keys())
        fig = make_subplots(2, n_feat, subplot_titles=key_names[:n_feat])

        patch_dist = patch_np[:, :, feature_pairs['patch_dist']].reshape((patch_np.shape[0], patch_np.shape[1]))
        patch_dist = np.round(patch_dist, 2)
        for col_i in range(n_feat):
            for row_i, pair_i in enumerate(feature_pairs[key_names[col_i]]):
                patch_i = patch_np[:, :, pair_i]
                if attn is not None:
                    mask = (attn>0) * attn
                    patch_i = patch_i * mask

                customdata = np.stack([patch_resnames[:, :, row_i], patch_dist], axis=-1)

                fig.add_trace(go.Heatmap(
                        z=patch_i,
                        customdata=customdata,
                        hovertemplate='<b>Value:%{z:.3f}</b><br>Amino Acid:%{customdata[0]}; dist:%{customdata[1]}',
                        name='',
                        colorscale='RdBu',
                        zmid=0,
                        showscale=False,
                        showlegend=False
                    ), row_i + 1, col_i + 1
                )
        fig.update_layout(title_text=f'The interactive patch pair for {ppi}. Hover to see the value and corresponding amino acid name.')
        if html_path is not None:
            plotly.offline.plot(fig, filename=html_path)
        else:
            fig.show()
            
    def __len__(self):
        return self.grid.shape[0]
    
    def read_scaled(self, ppi, device):
        idx = self.ppi_to_idx[ppi]
        grid = torch.from_numpy(np.expand_dims(self.grid[idx], 0))
        return grid.to(device)
    
    def __getitem__(self, idx):
        return self.grid[idx]