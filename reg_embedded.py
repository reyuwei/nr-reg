import sys
from pathlib import Path
import os
folder_root = Path().resolve()
sys.path.append(str(folder_root))
import pytorch3d.io
from tqdm import tqdm
import numpy as np
import torch
from loss.loss import NonRigidLayer
from lib.utils import get_lr, sum_dict
import time
from torch.utils.tensorboard import SummaryWriter
from loss.geodesic_distance import geodesic_distance
import argparse
import yaml

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

    template_mesh = pytorch3d.io.load_objs_as_meshes([args.template_mesh_path])
    template_mesh_gd = geodesic_distance(template_mesh.verts_packed().squeeze(), template_mesh.faces_packed().squeeze(), norm=False, num_workers=-1)
    target_mesh = pytorch3d.io.load_objs_as_meshes([args.target_mesh_path])

    ### create layer
    current_v = template_mesh.verts_packed()
    target_mesh = target_mesh.to(args.device)

    ## save result
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

        nglayer = NonRigidLayer(args, current_v, template_mesh.faces_packed(), template_mesh_gd, current_v.device)

        nglayer.compute_node(node_inter)

        node_trans = torch.zeros_like(nglayer.embed_node.detach()).to(nglayer.device).requires_grad_(True)
        node_optimizer = torch.optim.SGD([node_trans], lr=lr[ni], momentum=0.9)

        pbar = tqdm(range(iterations[ni]))
        for it in pbar:
            node_optimizer.zero_grad()
            loss_, _ = nglayer.embed_deform(None, node_trans, target_mesh, wts, sample_mesh_v[ni])
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

        new_v = nglayer.embed_deform(None, node_trans, target_mesh, wts, 1, forward_only=True)
        nglayer.update_v(new_v)
        nglayer.save_mesh(os.path.join(savepath_folder, "reg_nonrigid_{:d}.obj".format(node_inter)))
        current_v = new_v.detach()

        if ni == len(node_interval)-1:
            nglayer.save_mesh(os.path.join(savepath_folder, "reg_nonrigid_final.obj"))
