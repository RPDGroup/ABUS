import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
import argparse
from subprocess import Popen, PIPE
import shutil
from utils.utils import log_info, get_date


def read_config(args: argparse.Namespace) -> dict:
    """Read config file."""
    if not args.config: # default config file
        from config_default import make_config
        config = make_config()
    else:
        config_module = SourceFileLoader("config", args.config).load_module()
        config = config_module.config

    log_info("Configuration parameters", config)
    return config


def rename_chains(pid: str, ch: str, chain_pdb_dir: str, reversed: bool=True) -> tuple:
    """Rename chains in pdb file."""
    chains_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    all_chains = chains_choices[:]
    if reversed:
        all_chains.reverse()
        
    name = f"{pid}_{ch}.pdb"
    pdb_target_path = os.path.join(chain_pdb_dir, name)
    new_chains = []
    chains_seen = []
    output_path = os.path.join(os.getcwd(), name)
    with open(output_path, 'w') as out:
        with open(pdb_target_path, 'r') as f:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] not in chains_seen:
                        new_chains.append(all_chains.pop())
                        chains_seen.append(line[21])
                    line[21] = new_chains[-1]
                    line = ''.join(line)
                out.write(line)
    return name, ''.join(new_chains)
    
    
def execute_hdock(pid1: str, ch1: str, pid2: str, ch2: str, new_ch1: str, 
                  new_ch2: str,  PDB_TARGET: str, PDB_LIGAND: str, dock_dir: str) -> str:
    """Execute hdock."""
    out_pid = f"{pid1}-{ch1}-{pid2}-{ch2}_{new_ch1}_{new_ch2}"
    out_file = os.path.join(dock_dir, f"{out_pid}.out")
    if not os.path.exists(out_file):
        args = ['hdock', PDB_TARGET, PDB_LIGAND, '-out', f"{out_pid}.out"]
        log_info("Docking Log", ' '.join(args))
        process = Popen(args, stdout=PIPE, stderr=PIPE, cwd=dock_dir)
        stdout, stderr = process.communicate()
        log_info("Docking Log", stdout)
        log_info("Docking Log", stderr)
    else:
        log_info("Docking Log", f"{out_pid}.out already exists.")
    return out_pid


def run_hdock_one(ppi: str, pid1: str, ch1: str, pid2: str, ch2: str, dock_dir: str, config: dict) -> tuple:
    """Run docking."""
    if not os.path.exists(dock_dir):
        os.makedirs(dock_dir)
        
    log_info("Data Prepare", f"Start docking {pid1}_{ch1} and {pid2}_{ch2}.")
    curr_dir = os.getcwd()
    os.chdir(dock_dir)
    
    PDB_TARGET, new_ch1 = rename_chains(pid1, ch1, os.path.join(config['dirs']['chains_pdb'], ppi))
    PDB_LIGAND, new_ch2 = rename_chains(pid2, ch2, os.path.join(config['dirs']['chains_pdb'], ppi), reversed=False)
    
    out_pid = execute_hdock(pid1, ch1, pid2, ch2, new_ch1, new_ch2,  PDB_TARGET, PDB_LIGAND, dock_dir)
    
    if not os.path.exists(f"{out_pid}.pdb"):
        args_pr = ['createpl', f"{out_pid}.out", f"{out_pid}.pdb", '-complex', '-nmax', '100']
        process = Popen(args_pr, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
    return out_pid, new_ch1, new_ch2


def fix_residue_numbers(ppi: str, config: dict) -> None:
    """Fix residue numbers in pdb file."""
    pid, ch1, ch2 = ppi.split('_')
    pdb_file = os.path.join(config['dirs']['protonated_pdb'], pid+".pdb")
    pdb_tmp_file = os.path.join(config['dirs']['protonated_pdb'], pid+"_tmp.pdb")
    shutil.copyfile(pdb_file, pdb_tmp_file)
    prev_resid = ''
    prev_resname = ''
    rename_flag = False
    all_latters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    letter_i = 0
    with open(pdb_file, 'w') as out:
        with open(pdb_tmp_file, 'r') as f:
            for line in f.readlines():
                if line[:4] == 'ATOM':
                    curr_resname = line[17:20]
                    curr_resid = line[22:26]
                    if curr_resid == prev_resid and curr_resname != prev_resname:
                        rename_flag=True
                        letter_i += 1
                    if curr_resid != prev_resid:
                        rename_flag = False
                        letter_i = -1
                    if rename_flag:
                        line_list = [ch for ch in line]
                        line_list[26] = all_latters[letter_i]
                        # shift right the rest
                        for i in range(len(curr_resid)):
                            line_list[25-i] = curr_resid[-i-1]
                        line = ''.join(line_list)
                    prev_resid = curr_resid
                    prev_resname = curr_resname

                out.write(line)


def reset_config(config: dict, dock_dir: str) -> dict:
    """Reset configuration."""
    config['dirs']['data_prepare'] = dock_dir
    
    for dir_key in config['dirs'].keys():
        if dir_key not in ['data_prepare', 'save_model']:
            old_dir = config['dirs'][dir_key]
            base_dir = old_dir.split('/')[-1] if old_dir[-1] != '/' else old_dir.split('/')[-2]
            config['dirs'][dir_key] = os.path.join(dock_dir, base_dir)

    for dir in config['dirs'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
    return config


def combine_pdb(pdb1: str, pdb2: str, out_pdb: str, pdb_dir: str) -> None:
    """Combine pdb files."""
    with open(pdb_dir+out_pdb, 'w') as out:
        for pdb_file in [pdb1, pdb2]:
            with open(pdb_dir+pdb_file, 'r') as f1:
                for line in f1.readlines():
                    line = line.strip('\n').strip('new').strip(' ')
                    out.write(line+'\n')
                    

def extract_model(pdb_file: str, out_pdb: str, i: int) -> None:
    """Extract model i from pdb file."""
    to_write=False
    with open(pdb_file, 'r') as f:
        with open(out_pdb, 'w') as out:
            for line in f.readlines():
                # if line[:6] == 'HEADER':
                #     continue
                if line[:6] == 'ENDMDL':
                    to_write=False
                if to_write: # if the right model
                    out.write(line)
                if line[:5] == 'MODEL':
                    if line.split(' ')[-1].strip('\n') == str(i):
                        to_write = True
                    else:
                        to_write = False
                        
            
def fill_opacity(ppi: str, config: dict) -> None:
    """Fill the opacity of pdb files to one."""
    # HDOCK do not produce opacity scores as output which causes errors in the MaSIF data preparee module
    pid, ch1, ch2 = ppi.split('_')
    
  
    with open(os.path.join(config['dirs']['protonated_pdb'], f"{pid}.pdb"), 'w') as out:
        with open(os.path.join(config['dirs']['protonated_pdb'], f"{pid}_tmp.pdb"), 'r') as f:
            for line in f.readlines():
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    #pdb.set_trace()
                    line = line[:55] + ' 1.00' + line[60:] + '\n'
                    # line[57:60] = '1.00'
                    # line = ''.join(line)
                out.write(line)
    os.remove(os.path.join(config['dirs']['protonated_pdb'], f"{pid}_tmp.pdb"))
    

def merge_chains(pdb_in: str, ch1: str, ch2: str, pdb_out: str) -> None:
    """Merge chains in pdb file."""
    # Chains from the first protein will be renamed to A, while second protein will be renamed to Z
    with open(pdb_in, 'r') as f:
        with open(pdb_out, 'w') as out:
            for line in f.readlines():
                if line[:6] == 'HEADER':
                    continue
                if line[:4] == 'ATOM' or line[:6] == 'HETATM':
                    line = [char for char in line]
                    if line[21] in ch1:
                        line[21] = 'Z'
                    elif line[21] in ch2:
                        line[21] = 'A'
                    line = ''.join(line)
                out.write(line)