import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
from utils.utils import log_info, Colour
import numpy as np
from sklearn.neighbors import KDTree
from scipy import ndimage
from Bio.PDB import PDBParser, DSSP
from data_prepare.map_patch_atom import map_patch_indices


def get_new_coord_patch(radius: int) -> np.ndarray:
    """Get new patch coordinates."""
    new_patch_coord = []
    for i in range(0, radius*2):
        for j in range(0, radius*2):
            new_patch_coord.append((i-radius,j-radius))
    return np.array(new_patch_coord)


def remove_comments(pdb_path: str, pdb_tmp_path: str) -> None:
    """Standartize PDB file by adding white spaces and making each line exactly 80 characters"""
    with open(pdb_path, 'r') as in_pdb:
        with open(pdb_tmp_path, 'w') as out:
            for line in in_pdb.readlines():
                if "USER" not in line:
                    newline = []
                    for i in range(80):
                        if i<len(line.strip('\n')):
                            newline.append(line[i])
                        else:
                            newline.append(' ')
                    if line[:4]=="ATOM" or line[:6]=="HETATM":
                        newline[77]=newline[13]
                        #pdb.set_trace()
                    out.write(''.join(newline)+'\n')
                    #out.write(line)
    return None


def polar_to_cartesian(rho: np.ndarray, theta: np.ndarray, rotate_theta: float = 0) -> tuple:
    """Convert polar coordinates to cartesian coordinates."""
    # Interpolate the polar coordinates into a d x d square, where d is the diameter of the patch
    # rotate_theta - rotate all coordinates on a constant angle (used to search for matching patches).
    cart_coord_x = np.zeros(rho.shape)
    cart_coord_y = np.zeros(rho.shape)

    for coord_i in range(0, rho.shape[0]):
        rho_coord = rho[coord_i]
        theta_coord = theta[coord_i]
        cart_coord_x[coord_i] = rho_coord*np.cos(theta_coord+rotate_theta)
        cart_coord_y[coord_i] = rho_coord*np.sin(theta_coord+rotate_theta)

    return cart_coord_x, cart_coord_y


def compute_patch_grid(x: np.ndarray, y: np.ndarray, input_feat: np.ndarray, radius: int, 
                       interpolate: bool = True, stringarray: bool = False) -> np.ndarray:
    """Compute patch grid."""
    old_coord = np.stack((x,y), axis=-1)
    if not stringarray:
        patch_grid = np.zeros((radius*2, radius*2, input_feat.shape[1])) # shape = (24 x 24 x 5)
    else:
        patch_grid = np.array(radius*2*[np.array(['x' for x in range(radius*2)], dtype=object)])
        patch_grid = np.expand_dims(patch_grid, axis=-1)

    for feature_i in range(0, patch_grid.shape[-1]):
        #print("[{}] Computing grid for feature {}".format(datetime.now(), feature_i))
        old_coord_patch = old_coord
        new_coord_patch = get_new_coord_patch(radius) # grid coordinates [-r, r]
        # map old coordinates to the new grid:
        kdt = KDTree(old_coord_patch)
        if interpolate:
            dist, indx_old = kdt.query(new_coord_patch, k=4) #interpolate across 4 nearest neighbors
        else:
            dist, indx_old = kdt.query(new_coord_patch, k=1)
        # Square the distances (as in the original pyflann)
        dist = np.square(dist)

        for grid_point_i in range(0, dist.shape[0]): # go over each coordinate in a new grid and interpolate the features
            x_new, y_new = new_coord_patch[grid_point_i] # coordinate in a new grid
            r_tmp = np.sqrt(x_new ** 2 + y_new ** 2) # length of the radius from the center to the new point

            # Because our grid has negative coordinates, we will shift then to have only positive coordinates:
            column_i = x_new + radius # row index of final patch grid
            row_i = - y_new + radius -1 # column index of final patch grid

            # If the point outside of patch -
            if r_tmp>radius:
                patch_grid[row_i][column_i][feature_i] = 0
                continue

            if dist[grid_point_i][0]==0: # if the coordinate is for the neighbor that doesn't exist
                neigh_index_i = indx_old[grid_point_i][0]
                if x_new == 0 and y_new == 0:
                    patch_grid[row_i][column_i][feature_i] = input_feat[0][feature_i] # if center point
                else:
                    patch_grid[row_i][column_i][feature_i] = input_feat[neigh_index_i][feature_i]
                continue

            dist_grid_point = dist[grid_point_i]
            result_grid_points = indx_old[grid_point_i] #points to interpolate
            dist_to_include = []
            result_to_include = [] # old index list
            # Several old coordinates can map to a one new grid coordinate. Remove the redundancy:
            for i, result_i in enumerate(result_grid_points):
                if result_i not in result_to_include:
                    result_to_include.append(result_i)
                    dist_to_include.append(dist_grid_point[i])

            if interpolate:
                total_dist = np.sum(1 / np.array(dist_to_include))
                interpolated_value = 0
                for i, result_old_i in enumerate(result_to_include):
                        interpolated_value += input_feat[result_old_i][feature_i] * (1/ dist_to_include[i])/total_dist
                patch_grid[row_i][column_i][feature_i] = interpolated_value
            else:
                try:
                    patch_grid[row_i][column_i][feature_i] = input_feat[result_grid_points[0]][feature_i]
                except IndexError:
                    patch_grid[row_i][column_i][feature_i] = 0
    return patch_grid


def compute_dssp(ppi: str, config: dict) -> dict:
    """Compute DSSP values."""
    pid, ch1, ch2 = ppi.split('_')

    tmp_dir = config['dirs']['tmp']
    pdb_path = os.path.join(config['dirs']['protonated_pdb'], f"{pid}.pdb")
    pdb_tmp_path = os.path.join(tmp_dir, f"{pid}.pdb")

    # remove hydrogens
    remove_comments(pdb_path, pdb_tmp_path)

    parser = PDBParser(QUIET=1)
    struct = parser.get_structure(pid, pdb_tmp_path)

    model = struct[0]

    dssp = DSSP(model, pdb_tmp_path, dssp='mkdssp')  # example of a key: ('A', (' ', 1147, ' '))

    # Remove temporary file
    os.remove(pdb_tmp_path)
    return dssp


def convert_dssp_to_feat(dssp: dict, names_grid: np.ndarray) -> np.ndarray:
    """Convert DSSP object into grid of features"""
    dssp_features = np.zeros((names_grid.shape[0], names_grid.shape[1], 1))

    for i in range(names_grid.shape[0]):
        for j in range(names_grid.shape[1]):
            curr_name = names_grid[i][j][0] # example A:107:HIS-1621:CD2
            if curr_name!=0:
                # Read the residue from the array of names of a patch pair
                fields = curr_name.split(':')
                chain, resid = fields[0], fields[1]

                # Construct a key based on the current residue from two proteins
                # key example: ('A', (' ', 219, ' '))
                for key_i in dssp.keys():
                    if key_i[0]==chain and key_i[1][1] == int(resid):
                        dssp_key =key_i

                try:
                    dssp_features_i = dssp[dssp_key]
                except:
                    dssp_features[i][j][0] = 0
                    continue

                # Relative ASA:
                try:
                    dssp_features[i][j][0] = dssp_features_i[3]
                except:
                    dssp_features[i][j][0] = 0
    return dssp_features


def read_patch(ppi: str, ch: str, config: dict) -> tuple:
    """Read patch data."""
    pid = ppi.split('_')[0]
    patch_dir = os.path.join(config['dirs']['patches'], ppi)

    rho = np.load(os.path.join(patch_dir, f"{pid}_{ch}_rho_wrt_center.npy"), allow_pickle=True)
    theta = np.load(os.path.join(patch_dir, f"{pid}_{ch}_theta_wrt_center.npy"), allow_pickle=True)
    input_feat = np.load(os.path.join(patch_dir, f"{pid}_{ch}_input_feat.npy"), allow_pickle=True)
    resnames = np.load(os.path.join(patch_dir, f"{pid}_{ch}_resnames.npy"), allow_pickle=True)
    resnames = np.expand_dims(resnames, axis=1)

    # Read 3D coordinates
    coord_3d = np.load(os.path.join(patch_dir, f"{pid}_{ch}_coordinates.npy"), allow_pickle=True)

    return rho, theta, input_feat, resnames, coord_3d


def convert_one_patch(ppi: str, config: dict) -> None:
    """Convert one patch to image."""
    pid, ch1, ch2 = ppi.split('_')
    radius = config['ppi_const']['patch_r']
    out_dir = os.path.join(config['dirs']['grid'], ppi)
    os.makedirs(out_dir, exist_ok=True)
    
    out_grid = os.path.join(out_dir, f"{pid}_{ch1}_{ch2}.npy")
    out_resnames = os.path.join(out_dir, f"{pid}_{ch1}_{ch2}_resnames.npy")
    
    p1_rho, p1_theta, p1_input_feat, p1_resnames, p1_coord_3d = read_patch(ppi, ch1, config)
    p2_rho, p2_theta, p2_input_feat, p2_resnames, p2_coord_3d = read_patch(ppi, ch2, config)

    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)
    p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta)
    
    p1target_patch_grid = compute_patch_grid(p1target_x, p1target_y, p1_input_feat, radius)  # (r, r, n_feat)
    p2_patch_grid = compute_patch_grid(p2_x, p2_y, p2_input_feat, radius)  # (r, r, n_feat)
    
    p1name_grid = compute_patch_grid(p1target_x, p1target_y, p1_resnames, radius, interpolate=False, stringarray=True)
    p2name_grid = compute_patch_grid(p2_x, p2_y, p2_resnames, radius, interpolate=False, stringarray=True)
    
    p1_coord_grid = compute_patch_grid(p1target_x, p1target_y, p1_coord_3d, radius)
    p2_coord_grid = compute_patch_grid(p2_x, p2_y, p2_coord_3d, radius)
    dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_coord_grid), axis=-1))
    log_info("Convert Log", f"Average distance between surfaces: {dist_grid.mean()}", Colour.BLUE)

    single_grid = np.concatenate([p1target_patch_grid, p2_patch_grid, np.expand_dims(dist_grid, axis=-1)], axis=-1)
    names_grid = np.concatenate([p1name_grid, p2name_grid], axis=-1)
    
    dssp = compute_dssp(ppi, config)
    dssp_grid_1 = convert_dssp_to_feat(dssp, p1name_grid)
    dssp_grid_2 = convert_dssp_to_feat(dssp, p2name_grid)
    single_grid = np.concatenate([single_grid, dssp_grid_1, dssp_grid_2], axis=-1)
    
    np.save(out_grid, single_grid)
    np.save(out_resnames, names_grid)
    

def convert_to_images(ppi: str, config: dict) -> None:
    """Convert patches to images."""
    log_info("Convert Log", f"Start converting patches to images for {ppi}.", Colour.BLUE)
    pid, ch1, ch2 = ppi.split('_')
    out_grid = os.path.join(config['dirs']['grid'], ppi, f"{pid}_{ch1}_{ch2}_grid.npy")
    if os.path.exists(out_grid):
        log_info("Convert Log", f"Grid file {out_grid} already exists. Skipping...", Colour.BLUE)
        return
    convert_one_patch(ppi, config)
    