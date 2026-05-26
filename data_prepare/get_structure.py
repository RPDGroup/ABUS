import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
import time
from Bio.PDB import *
import numpy as np
from masif.source.input_output.protonate import protonate
from tqdm import tqdm
from scipy.spatial import cKDTree
from utils.utils import log_info, Colour

def protonate_pdb(ppi: str, config: dict) -> None:
    """Protonate PDB structure."""
    pid = ppi.split('_')[0]
    raw_pdb_filename = os.path.join(config['dirs']['raw_pdb'], f"{pid}.pdb")
    if not os.path.exists(raw_pdb_filename):
        # pdbl = PDBList(server="https://files.rcsb.org")
        pdbl = PDBList(server="https://files.wwpdb.org")
        raw_pdb_filename = pdbl.retrieve_pdb_file(pid, pdir=config['dirs']['raw_pdb'], file_format='pdb')
    else:
        # Remove MODEL line
        tmp_filename = os.path.join(config['dirs']['raw_pdb'], f"{pid}_tmp.pdb")
        os.rename(raw_pdb_filename, tmp_filename)
        with open(raw_pdb_filename, 'w') as out:
            with open(tmp_filename, 'r') as f:
                for line in f:
                    if "MODEL" not in line:
                        out.write(line)
        os.remove(tmp_filename)
        
    # Protonate downloaded file
    protonated_file = os.path.join(config['dirs']['protonated_pdb'], f"{pid}.pdb")
    protonate(raw_pdb_filename, protonated_file)


def download(ppi_list: list, config: dict, to_write: str=None) -> list:
    """Download PDB structures."""
    start = time.time()
    log_info("Download Log", f"Start downloading {len(ppi_list)} PDB structures.", Colour.BLUE)
    
    processed_ppi = []
    for i in tqdm(range(len(ppi_list))):
        ppi = ppi_list[i]
        pid = ppi.split('_')[0]
        
        pdb_filename = os.path.join(config['dirs']['protonated_pdb'], f"{pid}.pdb")
        if not os.path.exists(pdb_filename):
            protonate_pdb(ppi, config)
        else:
            log_info("Download Log", f"PDB file {pid} already exists. Skipping...", Colour.BLUE)
            
        if os.path.exists(pdb_filename):
            processed_ppi.append(ppi)
            
    if to_write is not None:
        with open(to_write, 'w') as out:
            for ppi in processed_ppi:
                out.write(ppi+'\n')
    
    log_info("Download Log", f"Done with downloading {len(processed_ppi)}/{len(ppi_list)} PDB structures. Took {(time.time()-start)/60:.2f}min.", Colour.BLUE)
    return processed_ppi


def get_coord_dict(pid: str, pdb_file: str, chain: str) -> dict:
    """Get coordinate dictionary."""
    parser = PDBParser(QUIET=True)
    try:
        pdb_struct = parser.get_structure(pid, pdb_file)
    except ValueError:
        log_info("Crop Log", f"Error: PDB file {pid}. Skipping...", Colour.RED)
        return None
        
    RES_dict = {
        "atom_id": [],
        "res_id": [],
        "chain_id": [],
        "atom_coord": []
    }
    all_atom_res_chain_pairs = []
    for i, atom in enumerate(pdb_struct.get_atoms()):
        res_id = atom.parent.id[1]
        chain_id = atom.get_parent().get_parent().get_id()
        atom_coord = list(atom.get_coord())
        atom_id = atom.serial_number
        
        if chain_id in chain:
            if (atom_id, res_id, chain_id) not in all_atom_res_chain_pairs:
                all_atom_res_chain_pairs.append((atom_id, res_id, chain_id))
                RES_dict["atom_id"].append(atom_id)
                RES_dict["res_id"].append(res_id)
                RES_dict["chain_id"].append(chain_id)
                RES_dict["atom_coord"].append(atom_coord)
                
    return RES_dict


def extract_pdb_chain(pdb_full_file: str, pdb_chain_file: str, ch: str) -> None:
    """Extract PDB chain."""
    with open(pdb_full_file, 'r') as f:
        with open(pdb_chain_file, 'w') as out:
            for line in f.readlines():
                if (line[0:4]=='ATOM' or line[0:6]=='HETATM') and line[21] in ch:
                    out.write(line)


def crop_pdb_one(ppi: str, config: dict) -> None:
    """Crop PDB structure."""
    parts = ppi.split('_')
    pid, ch1, ch2 = parts
    
    pdb_file = os.path.join(config['dirs']['protonated_pdb'], f"{pid}.pdb")
    crop_r = config['ppi_const']['crop_r']
    contact_d = config['ppi_const']['contact_d']
    out_dir = os.path.join(config['dirs']['cropped_pdb'], ppi)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{pid}_{ch1}_{ch2}.pdb")
    
    if os.path.exists(out_file):
        log_info("Crop Log", f"PDB file {out_file} already exists. Skipping...", Colour.BLUE)
        return
    
    res_dict_1 = get_coord_dict(pid, pdb_file, ch1)
    res_dict_2 = get_coord_dict(pid, pdb_file, ch2)
    
    if len(res_dict_1['res_id']) == 0 and len(res_dict_2['res_id']) == 0:
        log_info("Crop Log", f"PDB file {pid} is empty.", Colour.RED)
        return

    pdb_tree_1 = cKDTree(res_dict_1['atom_coord'])
    all_dist, all_idx_1 = pdb_tree_1.query(res_dict_2['atom_coord'])
    
    contact_indx2 = np.where(all_dist < contact_d)
    contact_indx1 = np.unique(all_idx_1[contact_indx2])
    
    center_iface1 = np.mean(np.array(res_dict_1['atom_coord'])[contact_indx1], axis=0)
    center_iface2 = np.mean(np.array(res_dict_2['atom_coord'])[contact_indx2], axis=0)
    
    dist_func1 = lambda x: (center_iface1[0] - x[0]) ** 2 + (center_iface1[1] - x[1]) ** 2 + (center_iface1[2] - x[2]) ** 2
    dist_func2 = lambda x: (center_iface2[0] - x[0]) ** 2 + (center_iface2[1] - x[1]) ** 2 + (center_iface2[2] - x[2]) ** 2
    
    dist_to_center1 = np.array([dist_func1(xi) for xi in res_dict_1['atom_coord']])
    dist_to_center2 = np.array([dist_func2(xi) for xi in res_dict_2['atom_coord']])
    
    min_dist1 = min(dist_to_center1) 
    min_dist2 = min(dist_to_center2)
    
    res_to_include1_indx = np.where(dist_to_center1 < (crop_r ** 2 + min_dist1 ** 2))
    res_to_include2_indx = np.where(dist_to_center2 < (crop_r ** 2 + min_dist2 ** 2))
    
    res_to_include1 = [res_dict_1['chain_id'][i] + ':' + str(res_dict_1['res_id'][i]) for i in range(0, len(res_dict_1['chain_id']))]
    res_to_include1 = np.unique(np.array(res_to_include1)[res_to_include1_indx])
    
    res_to_include2 = [res_dict_2['chain_id'][i] + ':' + str(res_dict_2['res_id'][i]) for i in range(0, len(res_dict_2['chain_id']))]
    res_to_include2 = np.unique(np.array(res_to_include2)[res_to_include2_indx])
    
    res_to_include = np.append(res_to_include1, res_to_include2)
    
    with open(out_file, 'w') as out:
        with open(pdb_file, 'r') as f:
            for line in f.readlines():
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    ch = line[21]
                    res_id = int(line[22:26])
                    
                    if f"{ch}:{res_id}" in res_to_include:
                        out.write(line)
                else:
                    out.write(line)
    
    extract_pdb_chain(out_file, os.path.join(out_dir, f'{pid}_{ch1}.pdb'), ch1)
    extract_pdb_chain(out_file, os.path.join(out_dir, f'{pid}_{ch2}.pdb'), ch2)
