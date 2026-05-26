import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
from utils.utils import log_info, Colour
import numpy as np
import pandas as pd
from Bio.PDB import *
from scipy.spatial import cKDTree


def get_start_res(resid: np.ndarray, chain_id: np.ndarray) -> np.ndarray:
    chain_curr = ''
    start_res = []
    start_res_curr = resid[0]
    for i, res_i in enumerate(resid):
        if chain_id[i] != chain_curr:
            start_res_curr = res_i
            chain_curr = chain_id[i]
        start_res.append(start_res_curr)
    return np.array(start_res)


def map_patch_atom_one(ppi: str, ch: str, config: dict) -> None:
    """Map patch atoms to patches."""
    pid = ppi.split('_')[0]
    patch_dir = os.path.join(config['dirs']['patches'], ppi)
    pdb_chains_dir = os.path.join(config['dirs']['chains_pdb'], ppi)
    out_mappings_dir = patch_dir
    
    out_table = os.path.join(out_mappings_dir, f"{pid}_{ch}_map.csv")
    x_coord = np.load(os.path.join(patch_dir, f"{pid}_{ch}_X_all.npy"))
    y_coord = np.load(os.path.join(patch_dir, f"{pid}_{ch}_Y_all.npy"))
    z_coord = np.load(os.path.join(patch_dir, f"{pid}_{ch}_Z_all.npy"))
    patch_coord = np.column_stack((x_coord,y_coord,z_coord))
    iface_labels = np.load(os.path.join(patch_dir, f"{pid}_{ch}_iface_labels.npy"))
    
    pdb_path = os.path.join(pdb_chains_dir, f"{pid}_{ch}.pdb")
    parser = PDBParser()
    pdb_struct = parser.get_structure(f"{pid}_{ch}", pdb_path)
    
    # Get heavy atoms
    heavy_atoms=[]
    heavy_orig_map = {}
    k=0
    for i, atom in enumerate(pdb_struct.get_atoms()):
        tags = atom.parent.get_full_id()
        if atom.element!='H' and tags[3][0]==' ': # if heavy atom and not heteroatom
            heavy_orig_map[k]=i # map heavy atom index to original pdb index
            heavy_atoms.append(atom)
            k+=1
            
    atom_coord = np.array([list(atom.get_coord()) for atom in heavy_atoms])
    atom_names = np.array([atom.get_id() for atom in heavy_atoms])
    residue_id = np.array([atom.parent.id[1] for atom in heavy_atoms])
    residue_name = np.array([atom.parent.resname for atom in heavy_atoms])
    chain_id = np.array([atom.get_parent().get_parent().get_id() for atom in heavy_atoms])
    
    # get start residue
    start_res = get_start_res(residue_id, chain_id)

    #Create KD Tree
    pdb_tree = cKDTree(atom_coord)
    
    dist, idx = pdb_tree.query(patch_coord) #idx is the index of pdb heavy atoms that close to every patch from [0 to N patches]
    result_pdb_idx=[]
    for i in idx:
        result_pdb_idx.append(heavy_orig_map[i])
    result_pdb_idx = np.array(result_pdb_idx) #index in original pdb
    #Combine everything to a table:
    df = pd.DataFrame({
        "patch_ind": range(0, len(result_pdb_idx)),
        "atom_ind": result_pdb_idx,
        "res_ind": residue_id[idx],
        "atom_name": atom_names[idx],
        "residue_name": residue_name[idx],
        "chain_id": chain_id[idx],
        "dist": dist,
        "iface_label": iface_labels,
        "start_res": start_res[idx]
    })

    df.to_csv(out_table, index=False)
    

def map_patch_indices(ppi: str, ch: str, config: dict) -> None:
    """Map patch indices to residue names."""
    pid = ppi.split('_')[0]
    mapping_table = os.path.join(config['dirs']['patches'], ppi, f"{pid}_{ch}_map.csv")
    mapping_df = pd.read_csv(mapping_table)
    indices_np = np.load(os.path.join(config['dirs']['patches'], ppi, f"{pid}_{ch}_list_indices.npy"))
    out_map = os.path.join(config['dirs']['patches'], ppi, f"{pid}_{ch}_resnames")

    mapping_df = mapping_df[mapping_df['patch_ind'].isin(indices_np)]
    res_names = np.array(['x' for i in range(len(indices_np))], dtype=object)
    for i,patch_i in enumerate(indices_np):
        tmp_df = mapping_df[mapping_df['patch_ind'] == patch_i].reset_index(drop=True)
        res_name_i = f"{tmp_df.loc[0]['chain_id']}:{tmp_df.loc[0]['res_ind']}:{tmp_df.loc[0]['residue_name']}-{tmp_df.loc[0]['atom_ind']}:{tmp_df.loc[0]['atom_name']}"
        res_names[i] = res_name_i
    np.save(out_map, res_names)
    

def map_patch_atom(ppi: str, config: dict) -> None:
    """Map patch atoms to patches."""
    log_info("Mapping Log", f"Start mapping patch atoms to patches for {ppi}.", Colour.BLUE)
    pid, ch1, ch2 = ppi.split('_')
    map_patch_atom_one(ppi, ch1, config)
    map_patch_atom_one(ppi, ch2, config)
    map_patch_indices(ppi, ch1, config)
    map_patch_indices(ppi, ch2, config)