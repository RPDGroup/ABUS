import os


def make_config(data_dir: str=None) -> dict:
    """Make config file."""
    config = {}

    config['dirs'] = {}
    if data_dir is not None:
        config['dirs']['data_prepare'] = data_dir
    else:
        config['dirs']['data_prepare'] = os.path.join(os.getcwd(), 'data_preparation')
    config['dirs']['raw_pdb'] = os.path.join(config['dirs']['data_prepare'], '00-raw_pdbs')
    config['dirs']['protonated_pdb'] = os.path.join(config['dirs']['data_prepare'], '01-protonated_pdb')
    config['dirs']['cropped_pdb'] = os.path.join(config['dirs']['data_prepare'], '02-cropped_pdbs')
    config['dirs']['chains_pdb'] = os.path.join(config['dirs']['data_prepare'], '03-chains_pdbs')
    config['dirs']['surface_ply'] = os.path.join(config['dirs']['data_prepare'], '04-surface_ply')
    config['dirs']['patches'] = os.path.join(config['dirs']['data_prepare'], '05-patches')
    config['dirs']['grid'] = os.path.join(config['dirs']['data_prepare'], '06-grid')
    config['dirs']['docked'] = os.path.join(config['dirs']['data_prepare'], 'docked')

    config['dirs']['dl_models'] = os.path.join(os.getcwd(), 'save_model')
    config['dirs']['tmp'] = os.path.join(os.getcwd(), 'tmp')

    config['ppi_const'] = {}
    config['ppi_const']['contact_d'] = 5 # minimum distance between residues to be considered as "contact point"
    config['ppi_const']['surf_contact_r'] = 1 # minimum distance between two surface points to be considered as "contact point"
    config['ppi_const']['patch_r'] = 16
    config['ppi_const']['crop_r'] = config['ppi_const']['patch_r'] + 1 # radius to crop (in Angstroms)
    config['ppi_const']['points_in_patch'] = 400

    config['mesh'] = {}
    config['mesh']['mesh_res'] = 1.0 # resolution of the mesh

    # Create Directories
    for dir in config['dirs'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
            
    # DL parameters
    os.environ["TMP"] = config['dirs']['tmp']
    os.environ["TMPDIR"] = config['dirs']['tmp']
    os.environ["TEMP"] = config['dirs']['tmp']
    
    return config


