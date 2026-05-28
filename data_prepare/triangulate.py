import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
import pymesh
import numpy as np
from shutil import copyfile, rmtree
from utils.utils import log_info, Colour
from sklearn.neighbors import KDTree
from data_prepare.get_structure import extract_pdb_chain
from masif.source.default_config.masif_opts import masif_opts
from masif.source.triangulation.computeMSMS import computeMSMS
from masif.source.triangulation.fixmesh import fix_mesh
from masif.source.input_output.extractPDB import extractPDB
from masif.source.input_output.save_ply import save_ply
from masif.source.triangulation.computeHydrophobicity import computeHydrophobicity
from masif.source.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from masif.source.triangulation.computeAPBS import computeAPBS
from masif.source.triangulation.compute_normal import compute_normal

def triangulate_one(ppi: str, ch: str, config: dict, pdb_filename: str) -> None:
    """Triangulate one chain."""
    pid = ppi.split('_')[0]
    chains_pdb_dir = config['dirs']['chains_pdb']
    tmp_pdb_dir = os.path.join(chains_pdb_dir, f"{pid}_{ch}")
    if not os.path.exists(tmp_pdb_dir):
        os.mkdir(tmp_pdb_dir)
    
    out_filename = os.path.join(tmp_pdb_dir, f"{pid}_{ch}")
    extractPDB(pdb_filename, f"{out_filename}.pdb", ch)
    
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(f"{out_filename}.pdb", protonate=True)
    
    vertex_hbond = computeCharges(out_filename, vertices1, names1)
    vertex_hphobicity = computeHydrophobicity(names1)
    
    vertices2 = vertices1
    faces2 = faces1
    
    mesh = pymesh.form_mesh(vertices2, faces2)
    regular_mesh = fix_mesh(mesh, config['mesh']['mesh_res'])
    
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)
    vertex_charges = computeAPBS(regular_mesh.vertices, f"{out_filename}.pdb", out_filename)
    chain_dir = os.path.join(chains_pdb_dir, ppi)
    os.makedirs(chain_dir, exist_ok=True)
    extract_pdb_chain(os.path.join(config['dirs']['protonated_pdb'], f"{pid}.pdb"), os.path.join(chain_dir, f"{pid}_{ch}.pdb"), ch)
    rmtree(tmp_pdb_dir)
    
    iface = np.zeros(len(regular_mesh.vertices))
    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    mesh = pymesh.form_mesh(v3, f3)
    full_regular_mesh = mesh
    v3 = full_regular_mesh.vertices
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)
    assert (len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0

    outply = os.path.join(config['dirs']['surface_ply'], ppi, f"{pid}_{ch}.ply")
    save_ply(outply, regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal, charges=vertex_charges, 
             normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity, iface=iface)


def triangulate_single(ppi: str, config: dict, overwrite: bool=False) -> None:
    """Triangulate single PDB structure."""
    log_info("Triangulate Log", f"Start triangulating {ppi}.")
    pid, ch1, ch2 = ppi.split('_')
    
    out_dir = os.path.join(config['dirs']['surface_ply'], ppi)
    os.makedirs(out_dir, exist_ok=True)
    out_ply_1 = os.path.join(out_dir, f"{pid}_{ch1}.ply")
    out_ply_2 = os.path.join(out_dir, f"{pid}_{ch2}.ply")
    
    if not overwrite and os.path.exists(out_ply_1) and os.path.exists(out_ply_2) or os.path.exists(f"{config['dirs']['surface_ply']}/{ppi}.npy"):
        log_info("Triangulate Log", f"Triangulated structures already exist for {ppi}. Skipping...")
        return
    
    pdb_filename = os.path.join(config['dirs']['cropped_pdb'], ppi, f"{ppi}.pdb")
    triangulate_one(ppi, ch1, config, pdb_filename)
    triangulate_one(ppi, ch2, config, pdb_filename)