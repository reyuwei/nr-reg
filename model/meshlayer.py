import torch
import trimesh
import pytorch3d
import numpy as np
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes
from lib.hand_utils import rbf_weights
from lib.transforms import compute_transmat

from loss.loss_collecter import ARAP_reg_loss, p3d_chamfer_distance_with_filter, update_p3d_mesh_shape_prior_losses


class EmbeddedDeform(torch.nn.Module):
    def __init__(self, node_sigma=15, thres_corres=8):
        super(EmbeddedDeform, self).__init__()
        self.node_sigma = node_sigma
        self.thres_corres = thres_corres

    def node_based_deform_pts(self, pts, deform_nodes, deform_nodes_R, deform_nodes_t, pts_weights):
        # pts = pts.reshape(-1, 3)
        pts = pts.reshape(-1, pts.shape[-1]) # with normal
        pts_v = pts[:, :3]
        
        # N, _ = deform_nodes.squeeze().shape
        # V, _ = pts.shape
        # # compute transformation mat for each node
        # deform_nodes_rmat = batch_rodrigues(deform_nodes_R.reshape(-1, 3)) # N, 9
        # deform_nodes_mat = torch.cat([deform_nodes_rmat.reshape(N, 3, 3), deform_nodes_t.reshape(N, 3, 1)], dim=-1) # N, 3, 4
        deform_nodes_mat = compute_transmat(deform_nodes, deform_nodes_R, deform_nodes_t)

        # compute weighted transformation for each vertex
        # verts_deform_mat_old = torch.matmul(deform_nodes_mat.permute(1,2,0), src_verts_weights.permute(1, 0))
        verts_deform_mat = torch.einsum("vn,nij->vij", pts_weights, deform_nodes_mat) # V, N * N, 3,4 -> V, 3, 4

        # apply weighted transformation to verts
        ones = torch.ones([pts_v.shape[0], 1], dtype=pts_v.dtype, device=pts_v.device)
        src_verts_hom = torch.cat([pts_v, ones], axis=-1)  # [N, 4]
        src_verts_trans = torch.einsum("ni,nji->nj", src_verts_hom, verts_deform_mat)
        # src_verts_trans = src_verts_trans[:, :3]   

        return src_verts_trans

    def similarity_matrix(self, mat):
        # get the product x * y
        # here, y = x.t()
        r = torch.mm(mat, mat.t())
        # get the diagonal elements
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        # compute the distance matrix
        D = diag + diag.t() - 2*r
        return D.sqrt()

    def forward(self, points, node, node_rotation, node_offset, weights):
        # points: (, N, 6) - point with normal
        deformed_points =  self.node_based_deform_pts(points, node, node_rotation, node_offset, weights)
        return deformed_points



class MeshNonRigidNet():
    def __init__(self, args, init_v, init_f, init_gd, device):

        self.args = args

        self.init_v = init_v.to(device)
        self.init_f = init_f.to(device)

        self.init_gd = init_gd.to(device)

        self.reg_weight = torch.ones(init_v.shape[0]).to(device)              

        self.device = device

        self.nonrigid_net = EmbeddedDeform().to(device)

    def compute_node(self, node_distance=8):
        # uniform sample verts
        if node_distance == 0:
            self.embed_node = self.init_v.clone()
            self.embed_node_weight = torch.eye(self.embed_node.shape[0]).to(self.device)
        
        else:

            mesh_tri = trimesh.Trimesh(self.init_v.cpu().detach().numpy(), self.init_f.cpu().detach().numpy(), process=False)

            surface_area = mesh_tri.area
            node_count = int(surface_area / ((node_distance/2)**2 * np.pi)) # node2node distance around 8 mm

            node_pc = mesh_tri.as_open3d.sample_points_poisson_disk(node_count, init_factor=5, pcl=None)
            sample_nodes = np.asarray(node_pc.points)

            # compute weight
            weight = rbf_weights(mesh_tri.vertices, sample_nodes)
            weight = torch.from_numpy(weight).float().to(self.device)
            sample_nodes = torch.from_numpy(sample_nodes).float().to(self.device)


            self.embed_node = sample_nodes
            self.embed_node_weight = weight


    def update_v(self, new_v):
        self.init_v = new_v.squeeze()

    def save_mesh(self, fname):
        template_mesh_deform = Meshes([self.init_v], [self.init_f])
        pytorch3d.io.IO().save_mesh(template_mesh_deform, fname)
        return template_mesh_deform


    def align_rigid(self, target_lmk, savename):

        matrix, _, _ = trimesh.registration.procrustes(self.init_lmk.cpu().detach().numpy(), target_lmk.cpu().detach().numpy())
        hom = torch.cat([self.init_v, torch.ones(self.init_v.shape[0], 1).to(self.device)], dim=-1)
        trans_v = (torch.from_numpy(matrix).float().to(self.device) @ hom.T).T[:, :3]

        self.update_v(trans_v)
        self.save_mesh(savename)


    def embed_deform(self, node_rot, node_trans, target_mesh, wts, sample_mesh_v, forward_only=False):
        # points: (, N, 6) - point with normal

        deformed_v = self.nonrigid_net.forward(self.init_v, self.embed_node, node_rot, node_trans, self.embed_node_weight)

        if forward_only:
            return deformed_v
        else:
            loss = self.compute_loss(deformed_v, node_rot, node_trans, target_mesh, wts, sample_mesh_v)

            return loss, deformed_v


    def compute_loss(self, deformed_v, node_rot, node_trans, target_mesh, wts, sample_mesh_v):
        loss_dict = {}

        if "mse_v" in wts:
            if wts['mse_v'] > 0:
                # loss_dict['mse_v'] = wts['mse_v'] * torch.nn.functional.mse_loss(deformed_v[~self.eye_nose_mask], target_mesh.verts_packed()[~self.eye_nose_mask])
                loss_dict['mse_v'] = wts['mse_v'] * ((deformed_v[~self.eye_nose_mask] - target_mesh.verts_packed()[~self.eye_nose_mask])**2).sum(-1).mean()

        # adaptive smoothness 
        if wts['smooth'] > 0:
            if self.embed_node.shape[0] == self.init_v.shape[0]:
                node_arap_loss, mask = ARAP_reg_loss(self.embed_node, node_rot, node_trans, node_gd=self.init_gd, thres_corres=self.args.thres_corres, node_sigma=self.args.node_sigma)
            else:
                node_arap_loss, mask = ARAP_reg_loss(self.embed_node, node_rot, node_trans, thres_corres=self.args.thres_corres, node_sigma=self.args.node_sigma)
            arap_loss = node_arap_loss * self.reg_weight[mask]
            loss_dict['smooth'] = wts['smooth'] * arap_loss.mean()

        # data
        sample_trg_v, sample_trg_n = sample_points_from_meshes(target_mesh, sample_mesh_v, return_normals=True)
        sample_trg_v = sample_trg_v.squeeze()
        sample_trg_n = sample_trg_n.squeeze()
        
        tetmesh = Meshes(verts=deformed_v.unsqueeze(0), faces=self.init_f.unsqueeze(0))
        sample_v, sample_v_normal = pytorch3d.ops.sample_points_from_meshes(tetmesh, sample_mesh_v, return_normals=True)
        sample_src = torch.cat([sample_v, sample_v_normal], dim=-1).squeeze()

        cf_dis, cf_dis_normal  = p3d_chamfer_distance_with_filter(
            sample_src[:,:3].unsqueeze(0),
            sample_trg_v.unsqueeze(0),
            x_normals=sample_src[:,3:].unsqueeze(0),
            y_normals=sample_trg_n.unsqueeze(0),
            angle_filter=self.args.data_angle_filter,
            distance_filter=self.args.data_distance_filter,
            wx=self.args.data_wx, wy=self.args.data_wy)

        if wts['data'] > 0:
            loss_dict['data'] = cf_dis * wts['data']
        if wts['data_normal'] > 0:
            loss_dict['data_normal'] = cf_dis_normal * wts['data_normal']

        # pytorch3d smooth
        update_p3d_mesh_shape_prior_losses(deformed_v, self.init_f, loss_dict, wts)

        return loss_dict