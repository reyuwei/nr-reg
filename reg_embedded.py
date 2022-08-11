import sys
from pathlib import Path
import os
folder_root = Path().resolve()
sys.path.append(str(folder_root))
import pytorch3d.io
from tqdm import tqdm
import numpy as np
import torch
import trimesh
from loss import NonRigidLayer
from utils import get_lr, sum_dict, load_js_lmk, compute_template_lmk_bid
from pytorch3d.structures.meshes import Meshes
import time
from torch.utils.tensorboard import SummaryWriter

import argparse
import yaml
ignore_lmk_idx = [5]

if __name__ == "__main__":

    args = argparse.Namespace(**yaml.safe_load(open("config\\param.yml")))

    if int(args.gpu) >= 0:
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.set_device(int(args.gpu))
        # args.device = torch.device('cuda')
        a = torch.eye(1).cuda()
        args.device = a.device
    else:
        args.device = torch.device('cpu')

    torch.cuda.empty_cache()
    savepath_folder = args.savepath_folder

    wts = args.wts

    test_tet_mesh_tri = trimesh.load(args.reg_mask_ply, process=False)
    black_v = test_tet_mesh_tri.visual.vertex_colors
    masked_v = ((black_v[:,:3]**2).sum(-1) < 10)
    masked_v_th = torch.from_numpy(masked_v).to(args.device)

    rest_v_gd = np.load(args.rest_v_gd, allow_pickle=True)
    rest_v_gd = torch.from_numpy(rest_v_gd[9476:, 9476:]).float()

    target_mesh = pytorch3d.io.load_objs_as_meshes([args.target_mesh_path])
    template_mesh = pytorch3d.io.load_objs_as_meshes([args.template_mesh_path])

    target_mesh_lmk_full = load_js_lmk(args.target_lmk_path)
    template_mesh_lmk_full = load_js_lmk(args.template_lmk_path)
    target_mesh_lmk = np.stack([target_mesh_lmk_full[i] for i in range(target_mesh_lmk_full.shape[0]) if i not in ignore_lmk_idx])
    template_mesh_lmk = np.stack([template_mesh_lmk_full[i] for i in range(template_mesh_lmk_full.shape[0]) if i not in ignore_lmk_idx])

    ### create layer
    bmc, faceid = compute_template_lmk_bid(template_mesh, template_mesh_lmk)
    target_mesh = target_mesh.to(args.device)
    target_mesh_lmk = torch.from_numpy(target_mesh_lmk).float().to(args.device)
    current_v = template_mesh.verts_packed()


    timestamp = str(time.time()).replace(".", "_")
    savepath_folder = savepath_folder + "\\{:s}".format(timestamp)
    tf_writer = SummaryWriter(Path(savepath_folder))
    backup_yaml = os.path.join(savepath_folder, "param.yaml")
    args_dict = vars(args)
    try:
        args_dict.pop("device")
    except:
        pass
    yaml.safe_dump(args_dict, open(backup_yaml, "w"))


    node_interval = args.node_interval
    iterations = args.iterations
    lr = args.lr
    sample_mesh_v = args.sample_mesh_v
    for ni, node_inter in enumerate(node_interval):

        nglayer = NonRigidLayer(args, current_v, template_mesh.faces_packed(), bmc, faceid, rest_v_gd, masked_v_th, masked_v_th.device)

        if ni == 0:
            nglayer.align_rigid(target_mesh_lmk, os.path.join(savepath_folder, "deform_rigid.obj"))
        nglayer.compute_node(node_inter)

        node_trans = torch.zeros_like(nglayer.embed_node.detach()).to(nglayer.device).requires_grad_(True)
        node_optimizer = torch.optim.SGD([node_trans], lr=lr[ni], momentum=0.9)

        pbar = tqdm(range(iterations[ni]))
        for it in pbar:
            node_optimizer.zero_grad()
            loss_, _ = nglayer.embed_deform(None, node_trans, target_mesh, target_mesh_lmk, wts, sample_mesh_v[ni])
            loss_str = ", ".join(["{:s}: {:.4f}".format(key, loss_[key]) for key in loss_])
            pbar.set_description("[%d] %s, lr: %.5f" % (node_inter, loss_str, get_lr(node_optimizer)))
            loss = sum_dict(loss_)

            if tf_writer is not None:  
                for k in loss_:
                    tf_writer.add_scalar("Loss terms/{:d}_{:s}".format(node_inter, k), loss_[k].item()/wts[k], global_step=it)
                tf_writer.add_scalar(
                    "Loss/{:d}".format(node_inter), loss.item(), global_step=it)

            loss.backward()
            node_optimizer.step()

        new_v = nglayer.embed_deform(None, node_trans, target_mesh, target_mesh_lmk, wts, 1, forward_only=True)
        nglayer.update_v(new_v)
        nglayer.save_mesh(os.path.join(savepath_folder, "reg_nonrigid_{:d}.obj".format(node_inter)))
        current_v = new_v.detach()

        if ni == len(node_interval)-1:
            nglayer.save_mesh(os.path.join(savepath_folder, "reg_nonrigid_final.obj"))
