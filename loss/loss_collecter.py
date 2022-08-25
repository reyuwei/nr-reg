import torch
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import chamfer_distance
from lib.transforms import compute_transmat
from loss.contactloss import compute_contact_loss
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)


# Losses to smooth / regularize the mesh shape
def update_p3d_mesh_shape_prior_losses(deformed_muscle, muscle_f, loss, weight_dict):
    centroid = deformed_muscle[:,:3].mean(0)

    deformed_muscle_mesh_p3d = Meshes(verts=(deformed_muscle[:,:3] - centroid).unsqueeze(0), faces=muscle_f.unsqueeze(0))
    

    # the edge length of the predicted mesh
    if weight_dict['edge'] > 0:
        loss["edge"] = mesh_edge_loss(deformed_muscle_mesh_p3d) * weight_dict['edge']
    
    # mesh normal consistency
    if weight_dict['normal'] > 0:
        loss["normal"] = mesh_normal_consistency(deformed_muscle_mesh_p3d) * weight_dict['normal']
    
    # mesh laplacian smoothing
    if weight_dict['laplacian'] > 0:
        loss["laplacian"] = mesh_laplacian_smoothing(deformed_muscle_mesh_p3d, method="uniform") * weight_dict['laplacian']


def similarity_matrix(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r
    return D.sqrt()


from pytorch3d.loss.chamfer import _handle_pointcloud_input, _validate_chamfer_reduction_inputs
from pytorch3d.ops.knn import knn_gather, knn_points
import torch.nn.functional as F
def p3d_chamfer_distance_with_filter(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    angle_filter=None,
    distance_filter=None,
    weights=None,
    batch_reduction: str = "mean",
    point_reduction: str = "mean",
    wx = 1.0, 
    wy = 1.0,
):
    """
    Chamfer distance between two pointclouds x and y. 
    # squared

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - (
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - (
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    if return_normals and angle_filter is not None:
        cham_norm_thres = 1 - torch.cos(torch.deg2rad(torch.tensor(angle_filter, dtype=torch.float).to(x.device)))
        norm_x_mask = cham_norm_x > cham_norm_thres
        norm_y_mask = cham_norm_y > cham_norm_thres

        cham_norm_x[norm_x_mask] = 0.0
        cham_norm_y[norm_y_mask] = 0.0
        cham_x[norm_x_mask] = 0.0
        cham_y[norm_y_mask] = 0.0

        x_lengths = torch.sum(~norm_x_mask)
        y_lengths = torch.sum(~norm_y_mask)
    
    if distance_filter is not None:
        dis_x_mask = cham_x > distance_filter
        dis_y_mask = cham_y > distance_filter
        cham_x[dis_x_mask] = 0.0
        cham_y[dis_y_mask] = 0.0
        
        if return_normals:
            cham_norm_x[dis_x_mask] = 0.0
            cham_norm_y[dis_y_mask] = 0.0

        x_lengths = torch.sum(~dis_x_mask)
        y_lengths = torch.sum(~dis_y_mask)

    if point_reduction is not None:
        # Apply point reduction sum
        cham_x = cham_x.sum(1)  # (N,)
        cham_y = cham_y.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
            cham_norm_y = cham_norm_y.sum(1)  # (N,)
        if point_reduction == "mean":
            cham_x /= x_lengths
            cham_y /= y_lengths
            if return_normals:
                cham_norm_x /= x_lengths
                cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x*wx + cham_y*wy
    cham_normals = cham_norm_x*wx + cham_norm_y*wy if return_normals else None

    return cham_dist, cham_normals

def dis_to_weight(dismat, thres_corres, node_sigma):
    dismat[dismat==0] = 1e5
    dismat[dismat>thres_corres] = 1e5
    node_weight = torch.exp(-dismat / node_sigma)
    norm = torch.norm(node_weight, dim=1)
    norm_node_weight = node_weight / (norm + 1e-6)
    norm_node_weight[norm==0] = 0
    return norm_node_weight

def ARAP_reg_loss(node_position, deform_nodes_R, deform_nodes_t, node_gd=None, thres_corres=64, node_sigma=15, device_max_cdist=10000):
    
    node_count = node_position.shape[0]
    
    if node_count > device_max_cdist:
        mask = torch.tensor(torch.rand(device_max_cdist) * node_count, dtype=torch.long).to(node_position.device)
        mask = torch.unique(mask)
    else:
        mask = torch.arange(node_count, dtype=torch.long).to(node_position.device)

    # print(mask[:10])

    node_position = node_position[mask]
    deform_nodes_t = deform_nodes_t[mask]
    if deform_nodes_R is not None:
        deform_nodes_R = deform_nodes_R[mask]
    if node_gd is not None:
        node_gd = node_gd[mask][:, mask]

    deform_nodes_mat = compute_transmat(node_position, deform_nodes_R, deform_nodes_t)
    if node_gd is not None:
        # should be geodesic distance!!!
        # gdist sqrt
        dismat = node_gd
    else:
        dismat = similarity_matrix(node_position) ** 2
    norm_node_weight = dis_to_weight(dismat.detach(), thres_corres, node_sigma).detach()

    ones = torch.ones([node_position.shape[0], 1], dtype=node_position.dtype, device=node_position.device)
    node_position_hom = torch.cat([node_position, ones], axis=-1)  # [N, 4]

    node_deform_pos_mat = torch.einsum("ni,kji->nkj", node_position_hom, deform_nodes_mat) # N, N, 3

    tjpj = torch.diagonal(node_deform_pos_mat).T.unsqueeze(1)
    node_dis_mat_total = ((node_deform_pos_mat - tjpj)**2).sum(-1)

    per_node_reg = torch.sum(node_dis_mat_total * norm_node_weight, axis=1)
        
    return per_node_reg, mask


def bone_attach_loss(deformed_muscle, target_piano , attach_id, weight_dict):
    w = weight_dict['attach']
    attach_tp = deformed_muscle[attach_id[:, 0]][:, :3]
    attach_tg = target_piano[attach_id[:,1]][:, :3]
    attach_tg = torch.unique(attach_tg, dim=0)
    loss, _ = chamfer_distance(attach_tp.unsqueeze(0), attach_tg.unsqueeze(0), point_reduction="mean") # squared
    return loss * w


def correspondence_to_smpl_function(points, grid, d_grid=None):
    grid = grid.permute(0, 3, 2, 1)
    sz = points.shape
    points_ = points.reshape(1, -1, 3).unsqueeze(1).unsqueeze(1)
    feats = torch.nn.functional.grid_sample(grid.unsqueeze(0), points_, align_corners=True)
    feats = feats.squeeze(2).squeeze(2).view(-1, sz[0], sz[1]).permute(1, 2, 0)
    return feats

def tsdf_loss(deformed_muscle, target_tsdf, target_tsdf_scale_center, weight_dict):
    # target_tsdf: (4,N,N,N)
    # target_tsdf_scale_center: (1, 4)
    points_unit = deformed_muscle[:,:3] * target_tsdf_scale_center[0] + target_tsdf_scale_center[1:]
    closest_distance = correspondence_to_smpl_function(points_unit.unsqueeze(0), target_tsdf[-1:, ...]).squeeze()
    closest_p = correspondence_to_smpl_function(points_unit.unsqueeze(0), target_tsdf[:6, ...]).squeeze()

    closest_p_normal = closest_p[:,3:] 
    closest_p_normal = closest_p_normal / torch.norm(closest_p_normal, dim=1).reshape(-1,1)

    normal_angle = (deformed_muscle[:,3:]*closest_p_normal).sum(-1)
    nnmask = torch.isnan(normal_angle)

    # acute_angle_ = normal_angle > -0.8 # normal angle smaller than 90 degree
    scale_normal_angle = -normal_angle + 2
    # closest_distance *= scale_normal_angle

    # closest_distance = closest_distance[acute_angle_] ** 2
    closest_distance = closest_distance ** 2
    # return closest_distance[~nnmask].mean() * weight_dict['data']
    return closest_distance.mean()* weight_dict['tsdf']


def compute_collision_loss_with_mesh(av, bv, bf, lab, distance_mode, use_mean, contact_thres):
    attr_loss, penetr_loss = compute_contact_loss(
        av.unsqueeze(0),
        bv.unsqueeze(0), bf,
        bv.unsqueeze(0),
        distance_mode=distance_mode, 
        use_mean=use_mean,
        contact_thresh=contact_thres
        )

    contact_loss = lab * attr_loss + (1-lab)* penetr_loss
    
    return contact_loss

def compute_collision_loss_with_multiple_mesh(av, mesh_dict: dict, lab, distance_mode, use_mean, ignore, contact_thres):
    
    contact_loss = torch.tensor(0).float().to(av.device)
    contact_list = []
    
    for key in mesh_dict.keys():

        if key == ignore:
            continue

        bv = mesh_dict[key].verts_packed()
        bvf = mesh_dict[key].verts_normals_packed()
        bv = torch.cat([bv, bvf], dim=-1)
        bf = mesh_dict[key].faces_packed()

        per_muscle_contact = compute_collision_loss_with_mesh(av, bv, bf, lab, distance_mode, use_mean, contact_thres)

        if per_muscle_contact > 0: 
            contact_list.append(key)
            contact_loss += per_muscle_contact
    
    if len(contact_list) > 0:
        contact_loss_mean = contact_loss / len(contact_list)
    else:
        contact_loss_mean = 0

    return contact_loss_mean, contact_list


def collision_loss(muscle_name, deformed_muscle, computed_muscles, target_piano, target_piano_f, weight_dict, lab, distance_mode, use_mean, contact_list=[], contact_thres=16):
    collision_ = compute_collision_loss_with_mesh(deformed_muscle, target_piano, target_piano_f, lab, distance_mode, use_mean, contact_thres)

    if len(computed_muscles) > 0:
        if len(contact_list) == 0: # collide with all
            collision_with_other_muscle, contact_list = compute_collision_loss_with_multiple_mesh(deformed_muscle, computed_muscles, 
            lab, distance_mode, use_mean, muscle_name, contact_thres)
        else: # collide with confirm contact
            computed_contact_muscles = {key: computed_muscles[key] for key in contact_list}
            collision_with_other_muscle, contact_list = compute_collision_loss_with_multiple_mesh(deformed_muscle, computed_contact_muscles,
            lab, distance_mode, use_mean, muscle_name, contact_thres)
        
        collision_ += collision_with_other_muscle
    
    w = weight_dict['collision']
    return collision_ * w

