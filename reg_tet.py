import os
import sys
from pathlib import Path

folder_root = Path().resolve()
sys.path.append(str(folder_root))

from models.musclelayertrain import MuscleLayerOne
from models.bonelayertrain import BoneLayerOne

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
from lib.util import batch_to_tensor_device, smooth_mesh, sum_dict
from lib.torch_functions import get_lr
from loss.nonrigid import HandNonRigidLayer
from pytorch3d.structures.meshes import Meshes
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
    settings = {
        "gpu": 0,
        "hand_side": ['right', 'left'],
        # "hand_side": ['left'],
        "tissue_type": ['skin', 'muscle', 'bone'],
        # "tissue_type": ['skin', 'bone'],
        # "tissue_type": ['muscle'],
        "input_mesh_path": r"data\nii\output\1655313928_106257",
        "reg_target_mesh_path": r"data\nii\mesh",
        "save_folder": r"data\nii\output_nonrigid",
    }
    torch.cuda.set_device(settings['gpu'])
    device = torch.eye(1).cuda().device
    torch.cuda.empty_cache()
    print(settings)

    pio = pytorch3d.io.IO()

    save_folder = Path(settings["save_folder"])
    os.makedirs(save_folder, exist_ok=True)
    timestamp = str(time.time()).replace(".", "_")
    exp_folder = save_folder / "{:s}".format(timestamp)
    tf_writer = SummaryWriter(str(exp_folder))
    backup_yaml = os.path.join(exp_folder, "settings.yaml")
    yaml.safe_dump(settings, open(backup_yaml, "w"))


    for ttype in settings['tissue_type']:
        load_param = os.path.join("configs", "0_" + ttype + ".yml")
        print(load_param)
        args = argparse.Namespace(**yaml.safe_load(open(load_param)))
        wts = args.term_weight

        target_mesh_path = glob(settings['reg_target_mesh_path'] +  "\\*_{:s}_smooth_simp.obj".format(ttype))
        assert len(target_mesh_path) == 1
        target_mesh_path = target_mesh_path[0]
        target_mesh = pytorch3d.io.load_objs_as_meshes([target_mesh_path]).to(device)

        template_dict_ttype = load_template_dict(ttype, args.template_folder)
        if ttype != "bone":
            attach_dict = template_dict_ttype['LAYER_ATTACH_DIS_ID_DICT']
            tet_template_dict = template_dict_ttype['tet_dict']

        if ttype == "muscle":
            muscle_vmask = trimesh.load(os.path.join(args.template_folder, "template_muscle_visible_mask.ply"), process=False)
            v_color = muscle_vmask.visual.vertex_colors
            color_marker = np.array([0, 0, 0]).reshape(1, 3)
            valid_muscle_v = np.sqrt(np.sum((v_color[:,:3] - color_marker)**2, axis=-1))<50
            faces = muscle_vmask.faces
            new_faces = []
            for f in faces:
                keep = True
                for fi in f:
                    if not valid_muscle_v[fi]:
                        keep=False
                if keep is True:
                    new_faces.append(f)
            new_faces = np.stack(new_faces)
            # new_mesh = trimesh.Trimesh(muscle_vmask.vertices, new_faces, process=False)
            # new_mesh.show()
            tet_template_dict['new_face'] = new_faces

        tet_template_dict = batch_to_tensor_device(tet_template_dict, device)


        for handside in settings['hand_side']:
            template_mesh_path = os.path.join(settings['input_mesh_path'], handside, "1", ttype + "_0.obj")
            template_mesh_attachbone_path = os.path.join(settings['input_mesh_path'], handside, "1", "bone_0.obj")

            print(target_mesh_path)
            print(template_mesh_path)
            print(template_mesh_attachbone_path)
            print("====== begin ======")
            
            exp_folder_current = exp_folder / (handside + "_" + ttype)
            os.makedirs(exp_folder_current, exist_ok=True)
            # continue

            template_mesh = pytorch3d.io.load_objs_as_meshes([template_mesh_path]).to(device)
            template_mesh_attachbone = pytorch3d.io.load_objs_as_meshes([template_mesh_attachbone_path]).to(device)

            # define net
            if ttype == "bone":
                mlayer = BoneLayerOne(template_mesh, target_mesh, wts, device, 
                save_tmp=True, save_folder=exp_folder_current, args=args)
            else:
                mlayer = MuscleLayerOne(template_mesh, target_mesh, ttype, tet_template_dict, wts, attach_dict, handside,
                device, attach_bone=template_mesh_attachbone, save_tmp=True, save_folder=exp_folder_current,  args=args)

            # define optimizer
            tet_node_optimizer = torch.optim.SGD([mlayer.tet_node_offset], lr=args.lr, momentum=0.9)

            def save_mesh(suffix, savesmooth=False):
                deformed_muscle_meshes = mlayer.get_mesh()
                for mn in deformed_muscle_meshes:
                    savepath = exp_folder_current / "{:s}_{:s}.obj".format(mn, suffix)
                    pio.save_mesh(deformed_muscle_meshes[mn], savepath)
                    if savesmooth:
                        deformed_smooth = smooth_mesh(deformed_muscle_meshes[mn])
                        savepath_smooth = exp_folder_current / "{:s}_{:s}_smooth.obj".format(mn, suffix)
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
                    pbar.set_description("[%s-%s] [%d] %s, lr: %.5f" % (
                        ttype, handside, mlayer.iter_counter_tet, loss_str, get_lr(tet_node_optimizer)))
                    loss = sum_dict(loss_)

                    if tf_writer is not None:
                        for k in loss_:
                            tf_writer.add_scalar("Loss terms/{:s}_{:s}_{:s}".format(ttype,handside, k), loss_[
                                                k].item()/wts[k], global_step=mlayer.iter_counter_tet)
                        tf_writer.add_scalar(
                            "Loss/{:s}-{:s}".format(ttype, handside), loss.item(), global_step=mlayer.iter_counter_tet)
                    
                    if "self_collision" in loss_ and wts['data'] == 0:
                        if loss_['self_collision'].item() < 1e-3:
                            break

                    loss.backward()
                    tet_node_optimizer.step()

                if global_itr % 1 == 0:
                    save_mesh(str(global_itr), savesmooth=True)
                    print("====== end ======")
            

