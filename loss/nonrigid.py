# import torch
# import trimesh
# import numpy as np
# import pytorch3d
# import torch.nn.functional as F
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.ops.knn import knn_points
# from pytorch3d.structures.meshes import Meshes
# from pytorch3d.loss import (
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,)
# from pytorch3d.loss.chamfer import _handle_pointcloud_input, _validate_chamfer_reduction_inputs
# from pytorch3d.ops.knn import knn_gather, knn_points
# from lib.util import vertices2landmarks
# from models.nonrigid_net import NonRigidNet
# from loss.contactloss import self_collision_loss
# from loss.loss_collecter import collision_loss, compute_collision_loss_with_mesh, similarity_matrix, dis_to_weight


# def p3d_chamfer_distance_with_filter(
#     x,
#     y,
#     x_lengths=None,
#     y_lengths=None,
#     x_normals=None,
#     y_normals=None,
#     angle_filter=None,
#     distance_filter=None,
#     weights=None,
#     batch_reduction: str = "mean",
#     point_reduction: str = "mean",
#     wx=1.0,
#     wy=1.0):
#     """
#     Chamfer distance between two pointclouds x and y. 
#     # squared

#     Args:
#         x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
#             a batch of point clouds with at most P1 points in each batch element,
#             batch size N and feature dimension D.
#         y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
#             a batch of point clouds with at most P2 points in each batch element,
#             batch size N and feature dimension D.
#         x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
#             cloud in x.
#         y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
#             cloud in x.
#         x_normals: Optional FloatTensor of shape (N, P1, D).
#         y_normals: Optional FloatTensor of shape (N, P2, D).
#         weights: Optional FloatTensor of shape (N,) giving weights for
#             batch elements for reduction operation.
#         batch_reduction: Reduction operation to apply for the loss across the
#             batch, can be one of ["mean", "sum"] or None.
#         point_reduction: Reduction operation to apply for the loss across the
#             points, can be one of ["mean", "sum"] or None.

#     Returns:
#         2-element tuple containing

#         - **loss**: Tensor giving the reduced distance between the pointclouds
#           in x and the pointclouds in y.
#         - **loss_normals**: Tensor giving the reduced cosine distance of normals
#           between pointclouds in x and pointclouds in y. Returns None if
#           x_normals and y_normals are None.
#     """
#     _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

#     x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
#     y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

#     return_normals = x_normals is not None and y_normals is not None

#     N, P1, D = x.shape
#     P2 = y.shape[1]

#     # Check if inputs are heterogeneous and create a lengths mask.
#     is_x_heterogeneous = (x_lengths != P1).any()
#     is_y_heterogeneous = (y_lengths != P2).any()
#     x_mask = (
#         torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
#     )  # shape [N, P1]
#     y_mask = (
#         torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
#     )  # shape [N, P2]

#     if y.shape[0] != N or y.shape[2] != D:
#         raise ValueError("y does not have the correct shape.")
#     if weights is not None:
#         if weights.size(0) != N:
#             raise ValueError("weights must be of shape (N,).")
#         if not (weights >= 0).all():
#             raise ValueError("weights cannot be negative.")
#         if weights.sum() == 0.0:
#             weights = weights.view(N, 1)
#             if batch_reduction in ["mean", "sum"]:
#                 return (
#                     (x.sum((1, 2)) * weights).sum() * 0.0,
#                     (x.sum((1, 2)) * weights).sum() * 0.0,
#                 )
#             return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

#     cham_norm_x = x.new_zeros(())
#     cham_norm_y = x.new_zeros(())

#     x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
#     y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

#     cham_x = x_nn.dists[..., 0]  # (N, P1)
#     cham_y = y_nn.dists[..., 0]  # (N, P2)

#     if is_x_heterogeneous:
#         cham_x[x_mask] = 0.0
#     if is_y_heterogeneous:
#         cham_y[y_mask] = 0.0

#     if weights is not None:
#         cham_x *= weights.view(N, 1)
#         cham_y *= weights.view(N, 1)

#     if return_normals:
#         # Gather the normals using the indices and keep only value for k=0
#         x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
#         y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

#         cham_norm_x = 1 - (
#             F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
#         )
#         cham_norm_y = 1 - (
#             F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
#         )

#         if is_x_heterogeneous:
#             cham_norm_x[x_mask] = 0.0
#         if is_y_heterogeneous:
#             cham_norm_y[y_mask] = 0.0

#         if weights is not None:
#             cham_norm_x *= weights.view(N, 1)
#             cham_norm_y *= weights.view(N, 1)

#     if return_normals and angle_filter is not None:
#         cham_norm_thres = 1 - torch.cos(torch.deg2rad(torch.tensor(angle_filter, dtype=torch.float).to(x.device)))
#         norm_x_mask = cham_norm_x > cham_norm_thres
#         norm_y_mask = cham_norm_y > cham_norm_thres

#         cham_norm_x[norm_x_mask] = 0.0
#         cham_norm_y[norm_y_mask] = 0.0
#         cham_x[norm_x_mask] = 0.0
#         cham_y[norm_y_mask] = 0.0

#         x_lengths = torch.sum(~norm_x_mask)
#         y_lengths = torch.sum(~norm_y_mask)
    
#     if distance_filter is not None:
#         dis_x_mask = cham_x > distance_filter
#         dis_y_mask = cham_y > distance_filter
#         cham_x[dis_x_mask] = 0.0
#         cham_y[dis_y_mask] = 0.0
        
#         if return_normals:
#             cham_norm_x[dis_x_mask] = 0.0
#             cham_norm_y[dis_y_mask] = 0.0

#         x_lengths = torch.sum(~dis_x_mask)
#         y_lengths = torch.sum(~dis_y_mask)

#     if point_reduction is not None:
#         # Apply point reduction sum
#         cham_x = cham_x.sum(1)  # (N,)
#         cham_y = cham_y.sum(1)  # (N,)
#         if return_normals:
#             cham_norm_x = cham_norm_x.sum(1)  # (N,)
#             cham_norm_y = cham_norm_y.sum(1)  # (N,)
#         if point_reduction == "mean":
#             cham_x /= x_lengths
#             cham_y /= y_lengths
#             if return_normals:
#                 cham_norm_x /= x_lengths
#                 cham_norm_y /= y_lengths

#     if batch_reduction is not None:
#         # batch_reduction == "sum"
#         cham_x = cham_x.sum()
#         cham_y = cham_y.sum()
#         if return_normals:
#             cham_norm_x = cham_norm_x.sum()
#             cham_norm_y = cham_norm_y.sum()
#         if batch_reduction == "mean":
#             div = weights.sum() if weights is not None else N
#             cham_x /= div
#             cham_y /= div
#             if return_normals:
#                 cham_norm_x /= div
#                 cham_norm_y /= div

#     cham_dist = cham_x*wx + cham_y*wy
#     cham_normals = cham_norm_x*wx + cham_norm_y*wy if return_normals else None

#     return cham_dist, cham_normals

# def update_p3d_mesh_shape_prior_losses(deformed_muscle, muscle_f, loss, weight_dict):
#     centroid = deformed_muscle[:,:3].mean(0)

#     deformed_muscle_mesh_p3d = Meshes(verts=(deformed_muscle[:,:3] - centroid).unsqueeze(0), faces=muscle_f.unsqueeze(0))
    

#     # the edge length of the predicted mesh
#     if weight_dict['edge'] > 0:
#         loss["edge"] = mesh_edge_loss(deformed_muscle_mesh_p3d) * weight_dict['edge']
    
#     # mesh normal consistency
#     if weight_dict['normal'] > 0:
#         loss["normal"] = mesh_normal_consistency(deformed_muscle_mesh_p3d) * weight_dict['normal']
    
#     # mesh laplacian smoothing
#     if weight_dict['laplacian'] > 0:
#         loss["laplacian"] = mesh_laplacian_smoothing(deformed_muscle_mesh_p3d, method="uniform") * weight_dict['laplacian']


# def ARAP_reg_loss(node_position, deform_nodes_R, deform_nodes_t, node_gd=None, thres_corres=64, node_sigma=15, device_max_cdist=10000):
    
#     node_count = node_position.shape[0]
    
#     if node_count > device_max_cdist:
#         mask = torch.tensor(torch.rand(device_max_cdist) * node_count, dtype=torch.long).to(node_position.device)
#         mask = torch.unique(mask)
#     else:
#         mask = torch.arange(node_count, dtype=torch.long).to(node_position.device)

#     # print(mask[:10])

#     node_position = node_position[mask]
#     deform_nodes_t = deform_nodes_t[mask]
#     if deform_nodes_R is not None:
#         deform_nodes_R = deform_nodes_R[mask]
#     if node_gd is not None:
#         node_gd = node_gd[mask][:, mask]

#     deform_nodes_mat = NonRigidNet.compute_transmat(node_position, deform_nodes_R, deform_nodes_t)
#     if node_gd is not None:
#         # should be geodesic distance!!!
#         # gdist sqrt
#         dismat = node_gd
#     else:
#         dismat = similarity_matrix(node_position) ** 2
#     norm_node_weight = dis_to_weight(dismat.detach(), thres_corres, node_sigma).detach()

#     ones = torch.ones([node_position.shape[0], 1], dtype=node_position.dtype, device=node_position.device)
#     node_position_hom = torch.cat([node_position, ones], axis=-1)  # [N, 4]

#     node_deform_pos_mat = torch.einsum("ni,kji->nkj", node_position_hom, deform_nodes_mat) # N, N, 3

#     tjpj = torch.diagonal(node_deform_pos_mat).T.unsqueeze(1)
#     node_dis_mat_total = ((node_deform_pos_mat - tjpj)**2).sum(-1)

#     per_node_reg = torch.sum(node_dis_mat_total * norm_node_weight, axis=1)
        
#     return per_node_reg, mask





# def rbf_weights(volume, control_pts, control_pts_weights=None):
#     """
#     Compute 3D volume weight according to control points with rbf and "thin_plate" as kernel
#     if control_pts_weights is None, directly return control points influences

#     Parameters
#     ----------
#     volume : (np.array, [N,3]) volume points with unknown weights 
#     control_pts : (np.array, [M,3]) control points
#     control_pts_weights : (np.array, [M,K]) control point weights

#     Returns
#     ----------
#     volume_weights : np.array, [N,K]
#     """
#     from scipy.interpolate import Rbf

#     if control_pts_weights is None:
#         control_pts_weights = np.eye(control_pts.shape[0])

#     xyz = volume.reshape(-1, 3)
#     chunk = 50000
#     rbfi = Rbf(control_pts[:, 0], control_pts[:, 1], control_pts[:, 2], control_pts_weights, function="thin_plate", mode="N-D")
#     # rbfi = Rbf(pts[:, 0], pts[:, 1], pts[:, 2], weight, function="multiquadric", mode="N-D")
#     weight_volume = np.concatenate([rbfi(xyz[j:j + chunk, 0], xyz[j:j + chunk, 1], xyz[j:j + chunk, 2]) for j in
#                                     range(0, xyz.shape[0], chunk)], 0)
#     weight_volume[weight_volume < 0] = 0
#     weight_volume = weight_volume / np.sum(weight_volume, axis=1).reshape(-1, 1)
#     weight_volume = weight_volume.reshape(xyz.shape[0], -1)
#     return weight_volume

# class HeadNonRigidLayer():
#     def __init__(self, args, init_v, init_f, lmk_bmc, lmk_faceid, init_gd, init_reg_mask, device):

#         self.args = args

#         self.init_v = init_v.to(device)
#         self.init_f = init_f.to(device)

#         self.lmk_bmc = lmk_bmc.to(device)
#         self.lmk_faceid = lmk_faceid.to(device)

#         self.init_lmk = vertices2landmarks(self.init_v.unsqueeze(0), self.init_f, self.lmk_faceid, self.lmk_bmc).squeeze()

#         self.init_gd = init_gd.to(device)

#         self.reg_weight = torch.ones(init_v.shape[0]).to(device)              
#         self.reg_weight[init_reg_mask.to(device)] *= self.args.semantic_large_reg

#         self.eye_nose_mask = init_reg_mask.to(device)

#         self.device = device

#         self.nonrigid_net = NonRigidNet().to(device)

#     def compute_node(self, node_distance=8):
#         # uniform sample verts
#         if node_distance == 0:
#             self.embed_node = self.init_v.clone()
#             self.embed_node_weight = torch.eye(self.embed_node.shape[0]).to(self.device)
        
#         else:

#             mesh_tri = trimesh.Trimesh(self.init_v.cpu().detach().numpy(), self.init_f.cpu().detach().numpy(), process=False)

#             surface_area = mesh_tri.area
#             node_count = int(surface_area / ((node_distance/2)**2 * np.pi)) # node2node distance around 8 mm

#             node_pc = mesh_tri.as_open3d.sample_points_poisson_disk(node_count, init_factor=5, pcl=None)
#             sample_nodes = np.asarray(node_pc.points)

#             # compute weight
#             weight = rbf_weights(mesh_tri.vertices, sample_nodes)
#             weight = torch.from_numpy(weight).float().to(self.device)
#             sample_nodes = torch.from_numpy(sample_nodes).float().to(self.device)


#             self.embed_node = sample_nodes
#             self.embed_node_weight = weight


#     def update_v(self, new_v):
#         self.init_v = new_v.squeeze()
#         self.init_lmk = vertices2landmarks(self.init_v.unsqueeze(0), self.init_f, self.lmk_faceid, self.lmk_bmc).squeeze()

#     def save_mesh(self, fname):
#         template_mesh_deform = Meshes([self.init_v], [self.init_f])
#         pytorch3d.io.IO().save_mesh(template_mesh_deform, fname)
#         return template_mesh_deform


#     def align_rigid(self, target_lmk, savename):

#         matrix, _, _ = trimesh.registration.procrustes(self.init_lmk.cpu().detach().numpy(), target_lmk.cpu().detach().numpy())
#         hom = torch.cat([self.init_v, torch.ones(self.init_v.shape[0], 1).to(self.device)], dim=-1)
#         trans_v = (torch.from_numpy(matrix).float().to(self.device) @ hom.T).T[:, :3]

#         self.update_v(trans_v)
#         self.save_mesh(savename)


#     def embed_deform(self, node_rot, node_trans, target_mesh, target_mesh_lmk, wts, sample_mesh_v, forward_only=False):
#         # points: (, N, 6) - point with normal

#         deformed_v = self.nonrigid_net.forward(self.init_v, self.embed_node, node_rot, node_trans, self.embed_node_weight)

#         if forward_only:
#             return deformed_v
#         else:
#             loss = self.compute_loss(deformed_v, node_rot, node_trans, target_mesh, target_mesh_lmk, wts, sample_mesh_v)

#             return loss, deformed_v


#     def compute_loss(self, deformed_v, node_rot, node_trans, target_mesh, target_lmk, wts, sample_mesh_v):
#         loss_dict = {}

#         if "mse_v" in wts:
#             if wts['mse_v'] > 0:
#                 # loss_dict['mse_v'] = wts['mse_v'] * torch.nn.functional.mse_loss(deformed_v[~self.eye_nose_mask], target_mesh.verts_packed()[~self.eye_nose_mask])
#                 loss_dict['mse_v'] = wts['mse_v'] * ((deformed_v[~self.eye_nose_mask] - target_mesh.verts_packed()[~self.eye_nose_mask])**2).sum(-1).mean()

#         # adaptive smoothness 
#         if wts['smooth'] > 0:
#             if self.embed_node.shape[0] == self.init_v.shape[0]:
#                 node_arap_loss, mask = ARAP_reg_loss(self.embed_node, node_rot, node_trans, node_gd=self.init_gd, thres_corres=self.args.thres_corres, node_sigma=self.args.node_sigma)
#             else:
#                 node_arap_loss, mask = ARAP_reg_loss(self.embed_node, node_rot, node_trans, thres_corres=self.args.thres_corres, node_sigma=self.args.node_sigma)
#             arap_loss = node_arap_loss * self.reg_weight[mask]
#             loss_dict['smooth'] = wts['smooth'] * arap_loss.mean()

#         # lmk
#         if wts['lmk'] > 0:
#             deformed_lmk = vertices2landmarks(deformed_v.unsqueeze(0), self.init_f, self.lmk_faceid, self.lmk_bmc).squeeze()
#             loss_dict['lmk'] = wts['lmk'] * F.mse_loss(deformed_lmk, target_lmk)

#         # data
#         sample_trg_v, sample_trg_n = sample_points_from_meshes(target_mesh, sample_mesh_v, return_normals=True)
#         sample_trg_v = sample_trg_v.squeeze()
#         sample_trg_n = sample_trg_n.squeeze()
        
#         tetmesh = Meshes(verts=deformed_v.unsqueeze(0), faces=self.init_f.unsqueeze(0))
#         sample_v, sample_v_normal = pytorch3d.ops.sample_points_from_meshes(tetmesh, sample_mesh_v, return_normals=True)
#         sample_src = torch.cat([sample_v, sample_v_normal], dim=-1).squeeze()

#         cf_dis, cf_dis_normal  = p3d_chamfer_distance_with_filter(
#             sample_src[:,:3].unsqueeze(0),
#             sample_trg_v.unsqueeze(0),
#             x_normals=sample_src[:,3:].unsqueeze(0),
#             y_normals=sample_trg_n.unsqueeze(0),
#             angle_filter=self.args.data_angle_filter,
#             distance_filter=self.args.data_distance_filter,
#             wx=self.args.data_wx, wy=self.args.data_wy)

#         if wts['data'] > 0:
#             loss_dict['data'] = cf_dis * wts['data']
#         if wts['data_normal'] > 0:
#             loss_dict['data_normal'] = cf_dis_normal * wts['data_normal']

#         # pytorch3d smooth
#         update_p3d_mesh_shape_prior_losses(deformed_v, self.init_f, loss_dict, wts)

#         return loss_dict


# class HandNonRigidLayer():
#     def __init__(self, args, init_v, init_f, device):

#         self.args = args

#         self.init_v = init_v.to(device)
#         self.init_f = init_f.to(device)

#         self.reg_weight = torch.ones(init_v.shape[0]).to(device)              

#         self.device = device

#         self.nonrigid_net = NonRigidNet().to(device)

#     def compute_node(self, node_distance=8):
#         # uniform sample verts
#         if node_distance == 0:
#             self.embed_node = self.init_v.clone()
#             self.embed_node_weight = torch.eye(self.embed_node.shape[0]).to(self.device)
        
#         else:

#             mesh_tri = trimesh.Trimesh(self.init_v[:,:3].cpu().detach().numpy(), self.init_f.cpu().detach().numpy(), process=False)

#             surface_area = mesh_tri.area
#             node_count = int(surface_area / ((node_distance/2)**2 * np.pi)) # node2node distance around 8 mm

#             node_pc = mesh_tri.as_open3d.sample_points_poisson_disk(node_count, init_factor=5, pcl=None)
#             sample_nodes = np.asarray(node_pc.points)

#             # compute weight
#             weight = rbf_weights(mesh_tri.vertices, sample_nodes)
#             weight = torch.from_numpy(weight).float().to(self.device)
#             sample_nodes = torch.from_numpy(sample_nodes).float().to(self.device)


#             self.embed_node = sample_nodes
#             self.embed_node_weight = weight


#     def update_v(self, new_v):
#         self.init_v = new_v.squeeze()

#     def save_mesh(self, fname):
#         template_mesh_deform = Meshes([self.init_v[:,:3]], [self.init_f])
#         pytorch3d.io.IO().save_mesh(template_mesh_deform, fname)
#         return template_mesh_deform


#     def align_rigid(self, target_lmk, savename):

#         matrix, _, _ = trimesh.registration.procrustes(self.init_lmk.cpu().detach().numpy(), target_lmk.cpu().detach().numpy())
#         hom = torch.cat([self.init_v, torch.ones(self.init_v.shape[0], 1).to(self.device)], dim=-1)
#         trans_v = (torch.from_numpy(matrix).float().to(self.device) @ hom.T).T[:, :3]

#         self.update_v(trans_v)
#         self.save_mesh(savename)


#     def embed_deform(self, node_rot, node_trans, target_mesh, wts, sample_mesh_v, forward_only=False):
#         # points: (, N, 6) - point with normal

#         deformed_v = self.nonrigid_net.forward(self.init_v, self.embed_node, node_rot, node_trans, self.embed_node_weight)

#         if forward_only:
#             return deformed_v
#         else:
#             loss = self.compute_loss(deformed_v, node_rot, node_trans, target_mesh, wts, sample_mesh_v)

#             return loss, deformed_v


#     def compute_loss(self, deformed_v, node_rot, node_trans, target_mesh, wts, sample_mesh_v):
#         loss_dict = {}

#         if "mse_v" in wts:
#             if wts['mse_v'] > 0:
#                 # loss_dict['mse_v'] = wts['mse_v'] * torch.nn.functional.mse_loss(deformed_v[~self.eye_nose_mask], target_mesh.verts_packed()[~self.eye_nose_mask])
#                 loss_dict['mse_v'] = wts['mse_v'] * ((deformed_v - target_mesh.verts_packed())**2).sum(-1).mean()

#         # adaptive smoothness 
#         if wts['smooth'] > 0:
#             if self.embed_node.shape[0] == self.init_v.shape[0]:
#                 node_arap_loss, mask = ARAP_reg_loss(self.embed_node, node_rot, node_trans, node_gd=self.init_gd, thres_corres=self.args.thres_corres, node_sigma=self.args.node_sigma)
#             else:
#                 node_arap_loss, mask = ARAP_reg_loss(self.embed_node, node_rot, node_trans, thres_corres=self.args.thres_corres, node_sigma=self.args.node_sigma)
#             arap_loss = node_arap_loss * self.reg_weight[mask]
#             loss_dict['smooth'] = wts['smooth'] * arap_loss.mean()

#         # data
#         sample_trg_v, sample_trg_n = sample_points_from_meshes(target_mesh, sample_mesh_v, return_normals=True)
#         sample_trg_v = sample_trg_v.squeeze()
#         sample_trg_n = sample_trg_n.squeeze()
        
#         tetmesh = Meshes(verts=deformed_v[:,:3].unsqueeze(0), faces=self.init_f.unsqueeze(0))
#         sample_v, sample_v_normal = pytorch3d.ops.sample_points_from_meshes(tetmesh, sample_mesh_v, return_normals=True)
#         sample_src = torch.cat([sample_v, sample_v_normal], dim=-1).squeeze()

#         cf_dis, cf_dis_normal  = p3d_chamfer_distance_with_filter(
#             sample_src[:,:3].unsqueeze(0),
#             sample_trg_v.unsqueeze(0),
#             x_normals=sample_src[:,3:].unsqueeze(0),
#             y_normals=sample_trg_n.unsqueeze(0),
#             angle_filter=self.args.data_angle_filter,
#             distance_filter=self.args.data_distance_filter,
#             wx=self.args.data_wx, wy=self.args.data_wy)

#         if wts['data'] > 0:
#             loss_dict['data'] = cf_dis * wts['data']
#         if wts['data_normal'] > 0:
#             loss_dict['data_normal'] = cf_dis_normal * wts['data_normal']

#         if wts['collision'] > 0:
#             collision_ = compute_collision_loss_with_mesh(deformed_v, target_mesh.verts_packed(), target_mesh.faces_packed(), self.args.contact_lab, self.args.contact_distance_mode, self.args.contact_distance_use_mean, self.args.contact_thres)
#             loss_dict['collision'] = collision_ * wts['collision']
    
#         # if wts['self_collision'] > 0:
#         #     loss_dict['self_collision'] = self_collision_loss(deformed_v, deformed_v,
#         #     self.nh_obj_net.face, self.deformed_tet_fn, self.tet_v_rest.squeeze(), self.nh_obj_net.element.squeeze(), self.weight_dict, 
#         #     distance_mode=self.args.contact_distance_mode)

#         # pytorch3d smooth
#         update_p3d_mesh_shape_prior_losses(deformed_v, self.init_f, loss_dict, wts)

#         return loss_dict