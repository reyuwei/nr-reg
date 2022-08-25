import torch
import numpy as np
from pytorch3d.structures.meshes import Meshes


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out

def sum_dict(los, ignore=""):
    temp = 0
    for l in los:
        if l != ignore:
            temp += los[l]
    return temp



def batch_to_tensor_device(batch, device):
    # data_ds = {
    #         # 'scan_bone': scan_mesh_bone,
    #         'scan_muscle': np.concatenate([scan_mesh_muscle.vertices, scan_mesh_muscle.vertex_normals], axis=1).astype('float32'),
    #         'scan_muscle_f': scan_mesh_muscle.faces,
    #         # 'scan_muscle_normal': scan_mesh_muscle.vertex_normals.astype('float32'),
    #         'lbs_piano': np.concatenate([naive_lbs_bone.vertices, naive_lbs_bone.vertex_normals], axis=1).astype('float32'),
    #         # 'lbs_piano_normal': naive_lbs_bone.vertex_normals.astype('float32'),
    #         'lbs_muscle': naive_muscles,
    #         'lbs_muscle_f': naive_muscles_faces,
    #         # 'lbs_muscle_normal': naive_muscles_normals,
    #         'muscle_names': muscle_names,
    #         'lbs_muscle_nodes': naive_muscles_nodes,
    #         'lbs_muscle_nodes_offset': naive_muscles_nodes_offset,
    #         'lbs_muscle_nodes_rotation':naive_muscles_nodes_rotation,
    #         'lbs_muscle_weights': naive_muscles_weights,
    #         'name': str(DATA_PATH / scan_name),
    #         }

    def to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr
        if arr.dtype == np.int64:
            arr = torch.from_numpy(arr)
        else:
            arr = torch.from_numpy(arr).float()
        return arr

    for key in batch:
        if isinstance(batch[key], np.ndarray):
            batch[key] = to_tensor(batch[key]).to(device)
        elif isinstance(batch[key], list):
            for i in range(len(batch[key])):
                if isinstance(batch[key][i], list):
                    for j in range(len(batch[key][i])):
                        if isinstance(batch[key][i][j], np.ndarray):
                            batch[key][i][j] = to_tensor(batch[key][i][j]).to(device)
                        elif isinstance(batch[key][i][j], Meshes):
                            batch[key][i][j] = batch[key][i][j].to(device)
                else:
                    batch[key][i] = to_tensor(batch[key][i]).to(device)
        elif isinstance(batch[key], str):
            continue
        elif isinstance(batch[key], dict):
            for key2 in batch[key].keys():
                batch[key][key2] = to_tensor(batch[key][key2]).to(device)
        elif isinstance(batch[key], Meshes):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        else:
            print(type(batch[key]), "Not Handled!!!")
    
    return batch
