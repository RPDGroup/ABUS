import os
import time
import torch
import shutil
import numpy as np
from tqdm import tqdm
from model.ABUS import ABUS
from utils.dataset import ABUS_Dataset
from torch.utils.data import DataLoader
from utils.utils import log_info, Colour
from model.ViT_pytorch import get_ml_config
from data_prepare.get_structure import download
from data_prepare.run_prepare import preprocess


default_params = {  
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
    'transformer_depth': 8
}


def construct_default_config(pdb_dir: str, out_dir: str) -> dict:
    """"""
    config = {}
    config['dirs'] = {}
    config['dirs']['raw_pdb'] = pdb_dir
    
    config['dirs']['data_prepare'] = os.path.join(out_dir, 'intermediate_files')
    config['dirs']['protonated_pdb'] = os.path.join(config['dirs']['data_prepare'], '01-protonated_pdb')
    config['dirs']['cropped_pdb'] = os.path.join(config['dirs']['data_prepare'], '02-cropped_pdbs')
    config['dirs']['chains_pdb'] = os.path.join(config['dirs']['data_prepare'], '03-chains_pdbs')
    config['dirs']['surface_ply'] = os.path.join(config['dirs']['data_prepare'], '04-surface_ply')
    config['dirs']['patches'] = os.path.join(config['dirs']['data_prepare'], '05-patches_16R')
    config['dirs']['grid'] = os.path.join(out_dir, 'grid_16R')
    
    config['dirs']['tmp'] = os.path.join(os.getcwd(), 'tmp')
    config['dirs']['vis'] = os.path.join(out_dir, 'patch_vis')
    
    config['ppi_const'] = {}
    config['ppi_const']['contact_d'] = 5  # minimum distance between residues to be considered as "contact point"
    config['ppi_const']['surf_contact_r'] = 1  # minimum distance between two surface points to be considered as "contact point"
    config['ppi_const']['patch_r'] = 16  # 16
    config['ppi_const']['crop_r'] = config['ppi_const']['patch_r'] + 1  # radius to crop (in Angstroms)
    config['ppi_const']['points_in_patch'] = 400  # 400 for 16 radius

    config['interact_feat'] = {}
    config['interact_feat']['atom_dist'] = True
    config['interact_feat']['dssp'] = True
    
    root_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['model'] = os.path.join(root_dir, 'save_model', 'test_model.pth')

    config['mesh'] = {}
    config['mesh']['mesh_res'] = 1.0  # resolution of the mesh
    
    # Create Directories
    for dir in config['dirs'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
            
    # DL parameters
    os.environ["TMP"] = config['dirs']['tmp']
    os.environ["TMPDIR"] = config['dirs']['tmp']
    os.environ["TEMP"] = config['dirs']['tmp']
    return config


def infer_from_model(ppi_list, grid_dir, model_path, params, device, radius, vis_dir):
    """Infer ABUS scores for a list of PPIs."""
    model_config = get_ml_config(params)
    model = ABUS(model_config, img_size=radius*2).float()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    model = model.to(device)
    log_info("Infer Log", f"Loaded ABUS model with {n_params} trainable parameters. Radius of the patch: {radius}A")

    dataset = ABUS_Dataset(grid_dir, ppi_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
    
    # Visualize patches
    for ppi in ppi_list:
        dataset.vis_patch(ppi, html_path=os.path.join(vis_dir, f"{ppi}.html"))

    all_outputs = []  # output score

    with torch.no_grad():
        start = time.time()
        for grid in tqdm(dataloader):
            grid = grid.to(device)
            output, _ = model(grid)
            all_outputs.append(output)

    output = torch.cat(all_outputs, axis=0)
    output = output.cpu().detach().numpy()
    return output
    

def infer_cmd(args):
    """Obtain ABUS scores"""
    if (not args.list and not args.ppi) or (args.list is not None and args.ppi is not None):
        raise AssertionError('Specify either "--list" or "--ppi" input')
    if args.list is not None:
        ppi_list = [x.strip('\n') for x in open(args.list)]
    elif args.ppi is not None:
        ppi_list = [args.ppi]
        
    out_dir = args.out_dir
    pdb_dir = args.pdb_dir
    log_info("Infer Log", f"Obtaining scores for {len(ppi_list)} complexes...")
    
    
    config = construct_default_config(pdb_dir, out_dir)
    download(ppi_list, config)
    preprocess(ppi_list, config)
    
    # Remove intermediate files
    # shutil.rmtree(config['dirs']['data_prepare'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = infer_from_model(
        ppi_list,
        grid_dir=config['dirs']['grid'],
        model_path=config['model'],
        params=default_params,
        device=device,
        radius=config['ppi_const']['patch_r'],
        vis_dir=config['dirs']['vis']
    )
    
    with open(os.path.join(out_dir, 'ABUS_scores.csv'), 'w') as out:
        out.write("PPI,ABUS_score\n")
        for i,ppi in enumerate(ppi_list):
            out.write(f"{ppi},{scores[i]}\n")
    log_info("Infer Log", f"ABUS is complete. See output in {out_dir}")
