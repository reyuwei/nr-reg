import sys
from pathlib import Path

folder_root = Path().resolve()
sys.path.append(str(folder_root))
from tqdm import tqdm
import torch
import trimesh
import numpy as np
from pytorch3d.structures.meshes import Meshes

from loss.contactutils import (
    batch_mesh_contains_points,
    batch_point_in_tet_barycoordinate,
    p3d_point_meshvf_face_distance
)


def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(inp.shape))
    ]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
    else:
        loss = torch.tensor(0).float().to(dists.device)
        # loss = torch.Tensor([0]).cuda()
    return loss

def masked_sum_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum()
    else:
        loss = torch.tensor(0).float().to(dists.device)
        # loss = torch.Tensor([0]).cuda()
    return loss


from pytorch3d.ops.knn import knn_points
def compute_contact_loss(
    floating_verts_pt_with_normal,
    steady_verts_pt_with_normal,
    steady_faces,
    steady_verts_pt_dense_with_normal,
    contact_thresh=1,
    collision_thresh=20,
    distance_mode="p2pl",
    use_mean=True,
    weight_mode="tanh",
    normal_filter=True
):
    # knn_obj2hand = knn_points(steady_verts_pt_dense, floating_verts_pt)
    # mins12, min12idxs = knn_obj2hand.dists.squeeze(-1), knn_obj2hand.idx.squeeze(-1)

    floating_verts_pt = floating_verts_pt_with_normal[..., :3]
    steady_verts_pt = steady_verts_pt_with_normal[..., :3]
    steady_verts_pt_dense = steady_verts_pt_dense_with_normal[..., :3]

    knn_hand2obj = knn_points(floating_verts_pt, steady_verts_pt_dense)
    mins21, min21idxs = knn_hand2obj.dists.squeeze(-1), knn_hand2obj.idx.squeeze(-1)

    # Get obj triangle positions
    obj_triangles = steady_verts_pt[:, steady_faces]
    exterior = batch_mesh_contains_points(
        floating_verts_pt.detach(), obj_triangles.detach()
    )
    # torch.cuda.empty_cache()
    penetr_mask = ~exterior # floating verts inside obj
    below_dist = mins21 < (contact_thresh ** 2)
    
    missed_mask = below_dist & exterior #  hand verts close to obj & hand verts outside obj
    missed_mask = missed_mask

    # early return
    if penetr_mask.sum() == 0:
        zero_loss = torch.tensor(0).float().to(floating_verts_pt.device)
        return zero_loss, zero_loss

    results_close = batch_index_select(steady_verts_pt_dense_with_normal, 1, min21idxs) # closest points on obj 
    anchor_dists = torch.norm(results_close[...,:3] - floating_verts_pt, 2, 2)
    # normal filter
    # discard_collision = torch.einsum("bnc, bnc -> bn", floating_verts_pt_with_normal[...,-3:], results_close[..., -3:]) > 0 # target normal similar
    direction = (results_close[...,:3] - floating_verts_pt) / anchor_dists.unsqueeze(-1)
    discard_collision = torch.einsum("bnc, bnc -> bn", floating_verts_pt_with_normal[...,-3:], direction) > 0 # target normal similar
    penetr_mask = penetr_mask & (~discard_collision)

    if distance_mode == "p2p":
        # Use squared distances to penalize contact
        contact_vals = ((results_close - floating_verts_pt) ** 2).sum(2) # anchor_dists ** 2 
        collision_vals = contact_vals
    elif distance_mode == "p2pl":
        # kaolin - sqrt - strongly wrong!!!
        # steady_mesh_kaolin = tm.from_tensors(steady_verts_pt.squeeze(), steady_faces)
        # contact_vals = point_to_surface_vec(floating_verts_pt.squeeze(), steady_mesh_kaolin)

        # p3d - squared
        contact_vals, _ = p3d_point_meshvf_face_distance(floating_verts_pt.squeeze(), steady_verts_pt.squeeze(), steady_faces)
        contact_vals = contact_vals.reshape(1, -1)
    
        # trimesh - sqrt
        # import trimesh
        # tri_steady_mesh = trimesh.Trimesh(steady_verts_pt.squeeze().cpu().detach().numpy(), steady_faces.cpu().detach().numpy())
        # contact_vals_tri = tri_steady_mesh.nearest.on_surface(floating_verts_pt.squeeze().cpu().detach().numpy())

        collision_vals = contact_vals
    else:
        raise ValueError(
            "distance_mode {} not in [p2p|p2pl]".format(distance_mode)
        )

    if weight_mode == "tanh":
        contact_vals = (contact_thresh**2) * torch.tanh(contact_vals/(contact_thresh**2))
        collision_vals = (collision_thresh**2) * torch.tanh(collision_vals/(collision_thresh**2))


    # Apply losses with correct mask
    if use_mean:
        missed_loss = masked_mean_loss(contact_vals, missed_mask)
        penetr_loss = masked_mean_loss(collision_vals, penetr_mask)
    else:
        missed_loss = masked_sum_loss(contact_vals, missed_mask)
        penetr_loss = masked_sum_loss(collision_vals, penetr_mask)

    # print('penetr_nb: {}'.format(penetr_mask.sum()))
    # print('missed_nb: {}'.format(missed_mask.sum()))
    max_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).max(1)[0].mean()
    )
    mean_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).mean(1).mean()
    )
    contact_info = {
        "attraction_masks": missed_mask,
        "repulsion_masks": penetr_mask,
        "contact_points": results_close,
        "min_dists": mins21,
    }
    metrics = {
        "max_penetr": max_penetr_depth,
        "mean_penetr": mean_penetr_depth,
    }
    # return missed_loss, penetr_loss, contact_info, metrics
    return missed_loss, penetr_loss


import numpy as np

def save_pointcloud(plist, name):
    import colorsys
    point_nb = plist.view(-1, 3).shape[0]
    if point_nb == 0:
        return
    color_list = []
    for i in range(point_nb):
        (r, g, b) = colorsys.hsv_to_rgb(i*1.0/point_nb, 0.8, 0.8)
        color_list.append([r*255, g*255, b*255, 255])
    color_list = np.stack(color_list)
    plist_nb = plist.view(-1, 3).cpu().detach().numpy()
    pc = np.hstack([plist_nb, color_list])
    np.savetxt("tmp\\col\\" + name+".obj", pc, fmt="v %5f %5f %5f %d %d %d %d")


def self_collision_loss(v, vn, f, fn, rest_v, e, weight_dict, distance_mode="p2pl"):
    # v.shape # torch.Size([5511, 3])
    # vn.shape # torch.Size([5511, 3])
    # f.shape # torch.Size([9514, 3])
    # fn.shape # torch.Size([9514, 3])
    # rest_v.shape # torch.Size([5511, 3])
    # e.shape # torch.Size([17490, 4])
    bn = 1
    point_nb = v.shape[0]
    triangle_nb = f.shape[0]
    element_nb = e.shape[0]

    obj_triangles = v[f]
    surface_v_mask = vn.sum(-1) != 0
    surface_v = v[surface_v_mask]
    surface_vn = vn[surface_v_mask]
    surface_point_nb = surface_v_mask.sum()

    mesh_tri = trimesh.Trimesh(v.view(-1, 3).detach().cpu().numpy(), f.view(-1, 3).detach().cpu().numpy())
    moving_surface_v = surface_v + surface_vn*1e-2
    _, index_ray = mesh_tri.ray.intersects_id(moving_surface_v.view(-1, 3).detach().cpu().numpy(), 
                                              surface_vn.view(-1, 3).detach().cpu().numpy(), 
                                              multiple_hits=True, return_locations=False)
    hits = np.zeros(surface_v.shape[0])
    unique, counts = np.unique(index_ray, return_counts=True)
    hits[unique] = counts
    exterior = (hits % 2 == 0)
    inside_mask = ~exterior


    # exterior, = batch_mesh_contains_points(
    #     surface_v.view(bn, surface_point_nb, 3).detach(), obj_triangles.view(bn, triangle_nb, 3, 3).detach(), 
    #     ray_directions=surface_vn.view(bn, surface_point_nb, 3).detach(), return_hit=False, 
    # )
    # torch.cuda.empty_cache()
    # inside_mask = ~exterior

    if inside_mask.sum() == 0:
        zero_loss = torch.tensor(0).float().to(v.device).requires_grad_(True)
        return zero_loss

    inside_v = surface_v[inside_mask]
    inside_vn = surface_vn[inside_mask]

    # find which tet and bary centric coordinate
    obj_tets = v[e]
    coo, tet_id, v_id = batch_point_in_tet_barycoordinate(inside_v.view(bn, -1, 3), obj_tets.view(bn, element_nb, 4, 3))

    if coo.shape[1] == 0:
        zero_loss = torch.tensor(0).float().to(v.device).requires_grad_(True)
        return zero_loss

    # compute rest position
    rest_obj_tets = rest_v[e].view(bn, element_nb, 4, 3)
    inside_v_rest = (coo.unsqueeze(-1) * rest_obj_tets[:, tet_id]).sum(-2) # bn, -1, 3
    inside_v = inside_v[v_id]
    inside_vn = inside_vn[v_id]
    # save_pointcloud(inside_v, "inside_v_raw")
    # save_pointcloud(inside_v_rest, "inside_v_rest")

    # find closest surface point
    if distance_mode == "p2p":
        knn_i2r = knn_points(inside_v_rest, rest_v[surface_v_mask].unsqueeze(0))
        mins21, min21idxs = knn_i2r.dists.squeeze(-1), knn_i2r.idx.squeeze(-1)
        inside_v_trg = v[surface_v_mask][min21idxs]
        inside_v_trg_n = vn[surface_v_mask][min21idxs]

        loss = ((v[surface_v_mask][inside_mask][v_id].unsqueeze(0) - inside_v_trg)**2).sum(-1) # squared

        # save_pointcloud(inside_v_trg, "inside_v_trg")
        # save_pointcloud(rest_v[surface_v_mask][min21idxs], "inside_v_rest_trg")

    elif distance_mode == "p2pl":
        contact_vals, fid = p3d_point_meshvf_face_distance(inside_v_rest.view(-1, 3), rest_v.view(point_nb, 3), f.view(triangle_nb, 3))
        inside_v_trg_n = fn[fid]
        loss = contact_vals.view(-1)
    else:
        raise ValueError(
            "distance_mode {} not in [p2p|p2pl]".format(distance_mode)
        ) 
    
    # filter
    a1 = (inside_v_trg_n * inside_vn).sum(-1) 
    keep_trg = a1 <= 0 # 反方向
    keep_trg = keep_trg & (contact_vals < 30) # 大于30mm的penetration不考虑

    if keep_trg.sum() == 0:
        zero_loss = torch.tensor(0).float().to(v.device).requires_grad_(True)
        return zero_loss

    loss = loss[keep_trg].mean() * weight_dict['self_collision']
    return loss




if __name__ == "__main__":
    import pytorch3d.io
    import numpy as np
    device = torch.zeros(1).cuda().device
    test_tet_mesh = pytorch3d.io.load_objs_as_meshes(["tmp\\col\\test_col.obj"])
    # test_tet_mesh = pytorch3d.io.load_objs_as_meshes([r"F:\OneDrive\Projects_ongoing\13_HANDMRI_Muscle\3_CODE_learn_muscle_skinning\loopreg\output\surface\1101_withs0\JB_SIMPLE_3\1635862956_1632817\skin_3mm_tet_0.obj"])
    tet_dict = np.load(r"F:\OneDrive\Projects_ongoing\13_HANDMRI_Muscle\3_CODE_learn_muscle_skinning\loopreg\assets_hand\template_final\obj_0918_3\surface\tetgen\dict.pkl", allow_pickle=True)
    e = torch.from_numpy(tet_dict['tet_e']['skin_3mm']).to(device)
    rest_v = torch.from_numpy(tet_dict['tet_v_rest']['skin_3mm']).float().to(device)
    rest_f = torch.from_numpy(tet_dict['tet_f']['skin_3mm']).to(device)
    init_tet = Meshes([rest_v], [rest_f])
    
    test_tet_mesh = test_tet_mesh.to(device)
    v = test_tet_mesh.verts_packed()
    vn = test_tet_mesh.verts_normals_packed()
    f = test_tet_mesh.faces_packed()
    fn = test_tet_mesh.faces_normals_packed()

    rest_v = init_tet.verts_packed()
    rest_vn = init_tet.verts_normals_packed()

    # v_offset = torch.rand(v.shape).to(device).requires_grad_(True)
    v_offset = torch.zeros_like(v).requires_grad_(True)

    torch.cuda.empty_cache()

    weight_dict = {"self_collision": 1.0}

    optimizer = torch.optim.SGD([v_offset], lr=0.5, momentum=0.9)
    # optimizer = torch.optim.LBFGS([v_offset], lr=0.2, max_iter=100, history_size=10, line_search_fn='strong_wolfe')
    pbar = tqdm(range(50))
    for it in pbar:
        # def closure():
        optimizer.zero_grad()
        col_loss = self_collision_loss(v+v_offset, vn, f, fn, rest_v, e, weight_dict, "p2pl")
        loss = col_loss

        # if loss.item() == 0: 
        #     break

        # if it % 5 == 0:
        new_mesh = Meshes([v+v_offset], [f])

        # pytorch3d.io.IO().save_mesh(new_mesh, "tmp\\col\\de_coll_{:d}.obj".format(it))

        vn = new_mesh.verts_normals_packed()
        fn = new_mesh.faces_normals_packed()

        pbar.set_description("%s" % (loss.item()))
        # loss.backward(retain_graph=True)
        loss.backward()
        # return loss
        optimizer.step()
        # optimizer.step(closure)

    new_mesh = Meshes([v+v_offset], [f])
    pytorch3d.io.IO().save_mesh(new_mesh, "tmp\\col\\de_coll_final_p2pl_sgd.obj")
