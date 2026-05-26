import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
import time
from utils.utils import log_info, Colour
from data_prepare.prepare_utils import *
from pdb2sql import StructureSimilarity
from data_prepare.get_structure import protonate_pdb
from data_prepare.get_structure import crop_pdb_one, download
from data_prepare.triangulate import triangulate_single
from data_prepare.compute_patches import compute_patches
from data_prepare.map_patch_atom import map_patch_atom
from data_prepare.convert_to_images import convert_to_images


def preprocess(processed_ppi: list, config: dict):
    """Preprocess PDB structures."""
    log_info("Data Prepare", f"Start preprocessing {len(processed_ppi)} PDB structures.")
    
    for ppi in processed_ppi:
        parts = ppi.split('_')
        
        grid_dir = os.path.join(config['dirs']['grid'], ppi)
        os.makedirs(grid_dir, exist_ok=True)
        out_grid = os.path.join(grid_dir, f"{ppi}.npy")
        
        if os.path.exists(out_grid):
            log_info("Data Prepare", f"Grid file {out_grid} already exists. Skip preprocessing.")
            continue
        try:
            crop_pdb_one(ppi, config)
            triangulate_single(ppi, config)
            compute_patches(ppi, config)
            map_patch_atom(ppi, config)
            convert_to_images(ppi, config)
        except Exception as e:
            log_info("Data Prepare", f"Error in preprocessing {ppi}. {e}", Colour.RED)
            download([ppi], config)
            fix_residue_numbers(ppi, config)
            crop_pdb_one(ppi, config)
            triangulate_single(ppi, config, overwrite=True)
            compute_patches(ppi, config, overwrite=True)
            map_patch_atom(ppi, config)
            convert_to_images(ppi, config)
            
        
def prepare_docking(processed_ppi: list, config: dict):
    """Prepare docking data."""
    log_info("Data Prepare", f"Start preparing docking data for {len(processed_ppi)} complexes.")
    for ppi in processed_ppi:
        pid, ch1, ch2 = ppi.split('_')
        dock_dir = os.path.join(config['dirs']['docked'], ppi)
        
        out_pid, new_ch1, new_ch2 = run_hdock_one(ppi, pid, ch1, pid, ch2, dock_dir, config)
        os.chdir(dock_dir)
        
        config = reset_config(config, dock_dir)
        log_info("Data Prepare", f"Reset configuration for {ppi}.")
        combine_pdb(f"{pid}_{ch1}.pdb", f"{pid}_{ch2}.pdb", "ref.pdb", './')
        log_info("Data Prepare", f"Combined {ppi} reference pdb files.")
        
        with open('irmsd.csv', 'w') as out:
            out.write('model_i,model_PPI,iRMSD,lRMSD,FNAT\n')
            for model_i in range(1, 101):  # generate each model

                try:
                    model_ppi = f"{pid}-model-{model_i}_{new_ch1}_{new_ch2}"
                    extract_model(f'{out_pid}.pdb', os.path.join(config['dirs']['protonated_pdb'], f'{pid}-model-{model_i}_tmp.pdb'), model_i)
                    fill_opacity(model_ppi, config)

                    # Pre-compute features for each model
                    if not os.path.exists(os.path.join(config['dirs']['grid'], model_ppi + '.npy')):
                        preprocess([model_ppi], config)

                    # Temporary create dimers (i.e. merge chains for a single protein, if more than one).
                    # Merging is necessary because FNAT calculation requires dimers.
                    curr_pdb_docked = os.path.join(config['dirs']['protonated_pdb'], f'{pid}-model-{model_i}.pdb')
                    curr_pdb_ref = os.path.join(config['dirs']['data_prepare'], 'ref.pdb')
                    tmp_pdb_docked = os.path.join(config['dirs']['tmp'], f'{pid}-model-{model_i}.pdb')
                    tmp_pdb_ref = os.path.join(config['dirs']['tmp'], f'ref_{pid}')

                    merge_chains(curr_pdb_docked, new_ch2, new_ch1, tmp_pdb_docked)
                    merge_chains(curr_pdb_ref, new_ch2, new_ch1, tmp_pdb_ref)

                    sim = StructureSimilarity(tmp_pdb_docked, tmp_pdb_ref)

                    irmsd = sim.compute_irmsd_pdb2sql(method='svd')
                    lrmsd = sim.compute_lrmsd_pdb2sql(method='svd')
                    fnat = sim.compute_fnat_pdb2sql()

                    os.remove(tmp_pdb_docked)
                    os.remove(tmp_pdb_ref)


                    # if irmsd < config['ppi_const']['iRMSD_threshold']:
                    #     shutil.move(os.path.join(config['dirs']['grid'], model_ppi + '.npy'), pos_dir + model_ppi + '.npy')
                    # else:
                    #     shutil.move(os.path.join(config['dirs']['grid'], model_ppi + '.npy'), neg_dir + model_ppi + '.npy')
                    out.write(f'{model_i},{model_ppi},{irmsd},{lrmsd},{fnat}\n')
                except Exception as e:
                    log_info("Data Prepare", f"Error in preparing {model_ppi}. {e}", Colour.RED)
        

def prepare(args):
    """Prepare data."""
    start_time = time.time()
    log_info("Data Prepare", "Start to prepare data.")
    
    if (not args.list and not args.ppi) or (args.list is not None and args.ppi is not None):
        raise AssertionError('Specify either "--list" or "--ppi" input')
    
    ppi_list = []
    if (args.list is not None):
        ppi_list = [x.strip('\n') for x in open(args.list)]
    elif (args.ppi is not None):
        ppi_list = [args.ppi]
    log_info("Data Prepare", f"Preprocessing {len(ppi_list)} complexes.")
    config = read_config(args)
    
    if not args.no_download: # download pdb structures
        processed_ppi = download(ppi_list, config)
    else:
        processed_ppi = ppi_list      
        
    if not args.download_only: # Need preprocess
        preprocess(processed_ppi, config)

    if args.prepare_docking: # prepare docking data
        prepare_docking(processed_ppi, config)  
        
    log_info("Data Prepare", f"Done with preparing data. Total execution time for data preparation: {(time.time()-start_time)/60:.2f}min.")
