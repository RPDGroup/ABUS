from utils.utils import set_environment
set_environment()

import argparse
from utils.utils import log_info
from utils.infer import infer_cmd
from data_prepare.run_prepare import prepare

parser = argparse.ArgumentParser(prog="ABUS v.1.0.0")
parser.add_argument('--config', help='config file')
sp = parser.add_subparsers()

# data prepare
sp_prepare = sp.add_parser('prepare', help='Data preparation module')
sp_prepare.add_argument('--list', help='List with PPIs in format PID_A_B')
sp_prepare.add_argument('--ppi', help='PPI in format PID_A_B (mutually exclusive with the --list option)')
sp_prepare.add_argument('--no_download',  default=False, action="store_true", help='If set True, the pipeline will skip the download part.')
sp_prepare.add_argument('--download_only',  default=False, action="store_true", help='If set True, the program will only download PDB structures without processing them.')
sp_prepare.add_argument('--prepare_docking',  default=False, action="store_true", help='If set True, re-dock ground truth structures and pre-process top 100 generated models')
sp_prepare.set_defaults(func=prepare)


# infer
sp_infer = sp.add_parser('infer', help='Inference module')
sp_infer.add_argument('--pdb_dir', required=True, help='Path to the PDB file of the complex that we need to score.')
sp_infer.add_argument('--list', help='Path to the list of protein complexes that we need to score. The list should contain the PPIs in the following format: PID_ch1_ch2, where PID is the name of PDB file, ch1 is the first chain(s) of the protein complex, and ch2 is the second chain(s). Ensure that PID does not contain an underscore')
sp_infer.add_argument('--ppi', help='PPI in format PID_A_B (mutually exclusive with the --list option)')
sp_infer.add_argument('--out_dir', required=True, help='Directory with output files.')
sp_infer.set_defaults(func=infer_cmd)

args = parser.parse_args()

if vars(args).get('func') is None:
    log_info("ERROR", "Please provide arguments (see the help message below).")
    parser.print_help()
    exit()

args.func(args) # run the function