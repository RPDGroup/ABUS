import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # root
from utils.utils import log_info, Colour
import time
import scipy
import pymesh
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from sklearn.neighbors import KDTree
from masif.source.masif_modules.read_data_from_surface import compute_ddc, normalize_electrostatics
from masif.source.geometry.compute_polar_coordinates import call_mds, compute_thetas, dict_to_sparse, compute_theta_all_fast


def get_iface_verticies(mesh: pymesh.Mesh) -> np.ndarray:
    """Get interface vertices from a mesh."""
    iface = mesh.get_attribute('vertex_iface')
    vertices = mesh.vertices
    iface_indx = np.where(iface > 0)
    if len(iface_indx[0])==0:
        log_info("Patch Log", "No interface found!")
        iface_indx = np.where(iface==0)
    return vertices[iface_indx]


def compute_theta_all(D: np.ndarray,  vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, 
                      idx: np.ndarray, radius: float, patch_center_i: int) -> np.ndarray:
    """Compute all angles for a patch."""
    mymds = MDS(n_components=2, n_init=1, max_iter=50, dissimilarity='precomputed', n_jobs=10)
    all_theta = []
    i = patch_center_i
    if i % 100 == 0:
        log_info("Patch Log", f"Processing patch {i}.")
    # Get the pairs of geodesic distances.

    neigh = D[i].nonzero()
    ii = np.where(D[i][neigh] < radius)[1]
    neigh_i = neigh[1][ii]
    pair_dist_i = D[neigh_i, :][:, neigh_i]
    pair_dist_i = pair_dist_i.todense()

    # Plane_i: the 2D plane for all neighbors of i
    plane_i = call_mds(mymds, pair_dist_i)

    # Compute the angles on the plane.
    theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)
    return theta


def compute_patch_center(mesh1: pymesh.Mesh, mesh2: pymesh.Mesh, radius: float) -> tuple:
    """Compute the center of the interaction between two meshes."""
    iface_vert1 = get_iface_verticies(mesh1)
    iface_vert2 = get_iface_verticies(mesh2)
    
    iface_vert_all = np.concatenate((iface_vert1, iface_vert2), axis=0)
    center_point = np.mean(iface_vert_all, axis=0)
    
    kdt1 = KDTree(mesh1.vertices)
    d, indx_cent1 = kdt1.query(np.expand_dims(center_point, axis=0))
    
    kdt2 = KDTree(mesh2.vertices)
    d, indx_cent2 = kdt2.query(np.expand_dims(center_point, axis=0))
    
    return center_point, indx_cent1[0][0], indx_cent2[0][0]


def compute_polar_coordinates(mesh: pymesh.Mesh, patch_center_i: int,  radius=12, max_vertices=200) -> np.ndarray:
    """Compute polar coordinates for a patch."""

    # Vertices, faces and normals
    vertices = mesh.vertices
    faces = mesh.faces
    norm1 = mesh.get_attribute('vertex_nx')
    norm2 = mesh.get_attribute('vertex_ny')
    norm3 = mesh.get_attribute('vertex_nz')
    normals = np.vstack([norm1, norm2, norm3]).T

    # Graph
    G = nx.Graph()
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))

    # Get edges
    f = np.array(mesh.faces, dtype=int)
    rowi = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]], axis=0)
    rowj = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]], axis=0)
    edges = np.stack([rowi, rowj]).T
    verts = mesh.vertices

    # Get weights
    edgew = verts[rowi] - verts[rowj]
    edgew = scipy.linalg.norm(edgew, axis=1)
    wedges = np.stack([rowi, rowj, edgew]).T

    G.add_weighted_edges_from(wedges)
    start = time.clock()

    dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius * 2)

    d2 = {}
    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.clock()
    log_info("Patch Log", f"Dijkstra took {(end - start):.2f}s.")
    D = dict_to_sparse(d2)

    # Compute the faces per vertex.
    idx = {}
    for ix, face in enumerate(mesh.faces):
        for i in range(3):
            if face[i] not in idx:
                idx[face[i]] = []
            idx[face[i]].append(ix)

    i = np.arange(D.shape[0])
    # Set diagonal elements to a very small value greater than zero..
    D[i, i] = 1e-8
    # Call MDS for all points.
    mds_start_t = time.clock()

    theta = compute_theta_all(D, vertices, faces, normals, idx, radius, patch_center_i)

    mds_end_t = time.clock()
    log_info("Patch Log", f"MDS took {(mds_end_t - mds_start_t):.2f}s.")

    n = len(d2)
    theta_out = np.zeros((max_vertices))
    rho_out = np.zeros((max_vertices))
    mask_out = np.zeros((max_vertices))

    i = patch_center_i
    # Assemble output.

    dists_i = d2[i]
    sorted_dists_i = sorted(dists_i.items(), key=lambda kv: kv[1])
    neigh = [int(x[0]) for x in sorted_dists_i[0:max_vertices]]
    rho_out[:len(neigh)] = np.squeeze(np.asarray(D[i, neigh].todense()))
    theta_out[:len(neigh)] = np.squeeze(theta[neigh])
    mask_out[:len(neigh)] = 1
    # have the angles between 0 and 2*pi
    theta_out[theta_out < 0] += 2 * np.pi

    return rho_out, theta_out, neigh, mask_out


def read_data_from_surface(ply_fn: str, patch_center_i: int, config: dict) -> tuple:
    """Read data from surface file."""
    mesh = pymesh.load_mesh(ply_fn)
    
    # Normals:
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1, n2, n3], axis=1)

    # Compute the angular and radial coordinates.
    radius = config['ppi_const']['patch_r']
    points_in_patch = config['ppi_const']['points_in_patch']
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, patch_center_i, radius=radius, max_vertices=points_in_patch)

    # Compute the principal curvature components for the shape index.
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)
    
    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob") / 4.5
    
    # Iface labels (for ground truth only)
    if "vertex_iface" in mesh.get_attribute_names():
        iface_labels = mesh.get_attribute("vertex_iface")
    else:
        iface_labels = np.zeros_like(hphob)

    # n: number of patches, equal to the number of vertices.
    n = len(mesh.vertices)

    input_feat = np.zeros((points_in_patch, 5))

    # Compute the input features for each patch.
    vix = patch_center_i
    # Patch members.
    neigh_vix = np.array(neigh_indices)

    # Compute the distance-dependent curvature for all neighbors of the patch.
    patch_v = mesh.vertices[neigh_vix]
    patch_n = normals[neigh_vix]
    patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
    mask_pos = np.where(mask == 1.0)[0]  # nonzero elements
    patch_rho = rho[mask_pos]  # nonzero elements of rho
    ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)

    # impute missing shape indicies with mean value for the whole patch
    si_patch = si[neigh_vix]

    si_patch = np.nan_to_num(si_patch, nan=np.nanmean(si_patch)) # replace nan values

    input_feat[:len(neigh_vix), 0] = si_patch
    input_feat[:len(neigh_vix), 1] = ddc
    input_feat[:len(neigh_vix), 2] = hbond[neigh_vix]
    input_feat[:len(neigh_vix), 3] = charge[neigh_vix]
    input_feat[:len(neigh_vix), 4] = hphob[neigh_vix]

    return input_feat, rho, theta, mask, neigh_indices, iface_labels, np.copy(mesh.vertices)


def save_precompute(ppi: str, ch: str, config: dict, input_feat: np.ndarray, rho: np.ndarray, theta: np.ndarray, 
                    mask: np.ndarray, neigh_indices: np.ndarray, iface_labels: np.ndarray, verts: np.ndarray, 
                    center_patch_i: int, patch_coord: np.ndarray) -> None:
    """Save precomputed patches."""
    pid = ppi.split('_')[0]
    # out_patch_dir = os.path.join(config['dirs']['patches'], ppi)
    my_precomp_dir = os.path.join(config['dirs']['patches'], ppi)
    if not os.path.exists(my_precomp_dir):
        os.mkdir(my_precomp_dir)

    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_rho_wrt_center.npy"), rho)
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_theta_wrt_center.npy"), theta)
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_input_feat.npy"), input_feat)    
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_mask.npy"), mask)
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_list_indices.npy"), neigh_indices)
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_iface_labels.npy"), iface_labels)
    # Save x, y, z
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_X.npy"), verts[center_patch_i, 0])
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_Y.npy"), verts[center_patch_i, 1])
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_Z.npy"), verts[center_patch_i, 2])

    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_X_all.npy"), verts[:, 0])
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_Y_all.npy"), verts[:, 1])
    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_Z_all.npy"), verts[:, 2])

    np.save(os.path.join(my_precomp_dir, f"{pid}_{ch}_coordinates.npy"), patch_coord)


def compute_patches(ppi: str, config: dict, overwrite: bool=False) -> None:
    """Compute patches for a PDB structure."""
    radius = config['ppi_const']['patch_r']
    log_info("Patch Log", f"Start computing patches for {ppi}.")
    
    pid, ch1, ch2 = ppi.split('_')
    out_feat1 = os.path.join(config['dirs']['patches'], ppi, f"{pid}_{ch1}_input_feat.npy")
    out_feat2 = os.path.join(config['dirs']['patches'], ppi, f"{pid}_{ch2}_input_feat.npy")
    if os.path.exists(out_feat1) and os.path.exists(out_feat2) and not overwrite:
        log_info("Patch Log", f"Patches for {ppi} already computed. Skipping...", Colour.BLUE)
        return
    
    ply_dir = os.path.join(config['dirs']['surface_ply'], ppi)
    ply_fn1 = os.path.join(ply_dir, f"{pid}_{ch1}.ply")
    ply_fn2 = os.path.join(ply_dir, f"{pid}_{ch2}.ply")
    
    mesh1 = pymesh.load_mesh(ply_fn1)
    mesh2 = pymesh.load_mesh(ply_fn2)
    
    patch_center, indx_c1, indx_c2 = compute_patch_center(mesh1, mesh2, radius)
    log_info("Patch Log", f"Center of the interaction: {patch_center}")
    
    input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1 = read_data_from_surface(ply_fn1, indx_c1, config)
    input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2 = read_data_from_surface(ply_fn2, indx_c2, config)

    points_in_patch = config['ppi_const']['points_in_patch']
    patch_coord1, patch_coord2 = np.zeros((points_in_patch, 3)), np.zeros((points_in_patch, 3))
    patch_coord1[:len(neigh_indices1)] = verts1[neigh_indices1]
    patch_coord2[:len(neigh_indices2)] = verts2[neigh_indices2]
    
    save_precompute(ppi, ch1, config, input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1, indx_c1, patch_coord1)
    save_precompute(ppi, ch2, config, input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2, indx_c2, patch_coord2)
