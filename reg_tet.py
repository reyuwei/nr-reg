import os
import sys
from pathlib import Path


folder_root = Path().resolve()
sys.path.append(str(folder_root))


import yaml
import time
import torch
import trimesh
import argparse
from glob import glob
import numpy as np
import pickle as pkl
import pytorch3d.io
from tqdm import tqdm
from lib.hand_utils import smooth_mesh
from model.tetlayer import TetNonRigidNet
from lib.torch_functions import batch_to_tensor_device, get_lr, sum_dict
from torch.utils.tensorboard import SummaryWriter



def load_template_dict(ttype, template_folder):

    template_dict = {}

    if ttype == "skin":
        template_dict['LAYER_ATTACH_DIS_ID_DICT'] = {0:[]}
        template_dict['tet_dict'] = np.load(os.path.join(template_folder, "skin_dict.pkl"), allow_pickle=True)
        tet_muscle_reg_pkl = os.path.join(template_folder, "tet_surface_reg_w_mask.pkl")
        if os.path.exists(tet_muscle_reg_pkl):
            tet_muscle_reg_mask = np.load(tet_muscle_reg_pkl, allow_pickle=True)
            template_dict['reg_mask'] = tet_muscle_reg_mask
    elif ttype=="muscle":
        template_dict['tet_dict'] = np.load(os.path.join(template_folder, "muscle_merge_dict.pkl"), allow_pickle=True)
        template_dict['LAYER_ATTACH_DIS_ID_DICT'] = np.load(os.path.join(template_folder, "naive_muscle_merge_attachment.pkl"), allow_pickle=True)

        tet_muscle_reg_pkl = os.path.join(template_folder, "tet_muscle_merge_reg_w_mask.pkl")
        if os.path.exists(tet_muscle_reg_pkl):
            tet_muscle_reg_mask = np.load(tet_muscle_reg_pkl, allow_pickle=True)
            template_dict['reg_mask'] = tet_muscle_reg_mask
    elif ttype=="bone":
        template_dict = {}
    else:
        print("no such ", ttype)

    return template_dict


if __name__ == "__main__":

    args = argparse.Namespace(**yaml.safe_load(open("config\\param_tet.yml")))

    if int(args.gpu) >= 0:
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.set_device(int(args.gpu))
        # args.device = torch.device('cuda')
        a = torch.eye(1).cuda()
        args.device = a.device
    else:
        args.device = torch.device('cpu')
    torch.cuda.empty_cache()
    ## save result
    timestamp = str(time.time()).replace(".", "_")
    savepath_folder = os.path.join(args.savepath_folder, timestamp)
    tf_writer = SummaryWriter(Path(savepath_folder))
    backup_yaml = os.path.join(savepath_folder, "param.yaml")
    args_dict = vars(args)
    try:
        args_dict.pop("device")
    except:
        pass
    yaml.safe_dump(args_dict, open(backup_yaml, "w"))

    pio = pytorch3d.io.IO()
    wts = args.term_weight
    exp_folder = Path(savepath_folder)


    print("====== begin ======")
    template_dict_ttype = load_template_dict("skin", args.template_folder)
    tet_template_dict = template_dict_ttype['tet_dict']
    tet_template_dict = batch_to_tensor_device(tet_template_dict, args.device)
    
    template_mesh = pytorch3d.io.load_objs_as_meshes([args.template_mesh_path]).to(args.device)
    target_mesh = pytorch3d.io.load_objs_as_meshes([args.target_mesh_path]).to(args.device)

    # define net
    mlayer = TetNonRigidNet(template_mesh, target_mesh, "skin", tet_template_dict, wts, None, "right", args.device, attach_bone=None, save_tmp=True, save_folder=exp_folder,  args=args)

    # define optimizer
    tet_node_optimizer = torch.optim.SGD([mlayer.tet_node_offset], lr=args.lr, momentum=0.9)

    def save_mesh(suffix, savesmooth=False):
        deformed_muscle_meshes = mlayer.get_mesh()
        for mn in deformed_muscle_meshes:
            savepath = exp_folder / "{:s}_{:s}.obj".format(mn, suffix)
            pio.save_mesh(deformed_muscle_meshes[mn], savepath)
            if savesmooth:
                deformed_smooth = smooth_mesh(deformed_muscle_meshes[mn])
                savepath_smooth = exp_folder / "{:s}_{:s}_smooth.obj".format(mn, suffix)
                pio.save_mesh(deformed_smooth, savepath_smooth)

    ### loop
    for global_itr in range(args.iterations):
        pbar = tqdm(range(args.sgd_iter))
        for it in pbar:
            tet_node_optimizer.zero_grad()

            if mlayer.iter_counter_tet % args.save_mesh_per_iter == 0:
                save_mesh(
                    "tmp_" + str(int(mlayer.iter_counter_tet)))

            loss_ = mlayer.optim_tet()
            loss_str = ", ".join(
                ["{:s}: {:.4f}".format(key, loss_[key]) for key in loss_])
            pbar.set_description("[%d] %s, lr: %.5f" % (
                mlayer.iter_counter_tet, loss_str, get_lr(tet_node_optimizer)))
            loss = sum_dict(loss_)

            if tf_writer is not None:
                for k in loss_:
                    tf_writer.add_scalar("Loss terms/{{:s}".format(k), loss_[
                                        k].item()/wts[k], global_step=mlayer.iter_counter_tet)
                tf_writer.add_scalar("Loss", loss.item(), global_step=mlayer.iter_counter_tet)
            
            if "self_collision" in loss_ and wts['data'] == 0:
                if loss_['self_collision'].item() < 1e-3:
                    break

            loss.backward()
            tet_node_optimizer.step()

        if global_itr % 1 == 0:
            save_mesh(str(global_itr), savesmooth=True)
            print("====== end ======")
    

