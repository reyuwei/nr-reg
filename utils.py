import torch
import json
import trimesh
import numpy as np


def load_js_lmk(filepath):
    lmk_js = json.load(open(filepath))
    lmk = [[one_lmk['x'], one_lmk['y'], one_lmk['z']] for one_lmk in lmk_js]
    lmk = np.stack(lmk)
    return lmk

def compute_template_lmk_bid(template_mesh, template_mesh_lmk):
    template_mesh_tri = trimesh.Trimesh(template_mesh.verts_packed(), template_mesh.faces_packed(), process=False)
    triangles = template_mesh_tri.triangles
    mesh_query = trimesh.proximity.ProximityQuery(template_mesh_tri)
    closest, distance, triangle_id = mesh_query.on_surface(template_mesh_lmk)
    bmc = trimesh.triangles.points_to_barycentric(triangles[triangle_id], template_mesh_lmk)

    bmc = torch.from_numpy(bmc).float()
    triangle_id = torch.from_numpy(triangle_id).long()

    return bmc, triangle_id

def vertices2landmarks(vertices,faces,lmk_faces_idx,lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    # lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        # batch_size, -1, 3)
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        1, -1, 3)
    lmk_faces = lmk_faces.repeat([batch_size,1,1])

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,lf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks




def sum_dict(los, ignore=""):
    temp = 0
    for l in los:
        if l != ignore:
            temp += los[l]
    return temp


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def dis_to_weight(dismat, thres_corres, node_sigma):
    dismat[dismat==0] = 1e5
    dismat[dismat>thres_corres] = 1e5
    node_weight = torch.exp(-dismat / node_sigma)
    norm = torch.norm(node_weight, dim=1)
    norm_node_weight = node_weight / (norm + 1e-6)
    norm_node_weight[norm==0] = 0
    return norm_node_weight


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

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                   2], norm_quat[:,
                                                       3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def compute_transmat(deform_nodes, deform_nodes_R, deform_nodes_t):

    device = deform_nodes.device
    N, _ = deform_nodes_t.squeeze().shape

    if deform_nodes_R is not None:
        # rotate around deform_nodes, then translate
        deform_nodes_rmat = batch_rodrigues(deform_nodes_R.reshape(-1, 3)) # N, 9
    else:
        deform_nodes_rmat = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(device).reshape(-1, 9)

    # move to zero
    trans_zero_mat = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
    trans_zero_mat_back = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)

    trans_zero_mat[:, :3, -1] = (-deform_nodes).reshape(-1, 3)
    trans_zero_mat_back[:, :3, -1] = deform_nodes.reshape(-1, 3)

    rots_mat = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
    rots_mat[:, :3, :3] = deform_nodes_rmat.reshape(N, 3, 3)
        
    trans_offset_mat = torch.eye(4).unsqueeze(0).repeat(N, 1, 1).to(device)
    trans_offset_mat[:, :3, -1] = deform_nodes_t.reshape(N, 3)

    deform_nodes_mat = trans_offset_mat @ trans_zero_mat_back @ rots_mat @ trans_zero_mat

    deform_nodes_mat = deform_nodes_mat[:, :3, :]

    # deform_nodes_mat = torch.cat([deform_nodes_rmat.reshape(N, 3, 3), deform_nodes_t.reshape(N, 3, 1)], dim=-1) # N, 3, 4
    return deform_nodes_mat



class NonRigidNet(torch.nn.Module):
    def __init__(self, node_sigma=15, thres_corres=8):
        super(NonRigidNet, self).__init__()
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
