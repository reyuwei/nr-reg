import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def normalizev(vec):
    if isinstance(vec, np.ndarray):
        return vec / np.linalg.norm(vec)
    else:
        return vec / torch.norm(vec)



def ComputeGlobalS(target_skeleton_batch, mano_skeleton_batch, isBody=False, isMinSke=False):
    batch_size = target_skeleton_batch.shape[0]
    if isBody:
        links = [
            (1, 8),
            # (2,3,4),
            # (5,6,7),
            # (10,11),
            # (13,14),
            (2, 5),
        ]
        bone_num = 2
        width_bone_num = 1
    elif isMinSke is False:  # hand only
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
            (5, 9, 13, 17),
        ]
        bone_num = 23
        width_bone_num = 3
    else:  # minimum skeleton only (0,9,17)
        links = [
            (0, 9),
            (0, 17),
            (9, 17)
        ]
        bone_num = 3
        width_bone_num = 1

    bone_ratios = torch.zeros(batch_size, bone_num)
    itr = 0
    for link in links:
        for j1, j2 in zip(link[0:-1], link[1:]):
            pred_bone_len = torch.norm(mano_skeleton_batch[:, j1, :] - mano_skeleton_batch[:, j2, :], dim=1)
            target_bone_len = torch.norm(target_skeleton_batch[:, j1, :] - target_skeleton_batch[:, j2, :], dim=1)
            ratio = target_bone_len / pred_bone_len
            bone_ratios[:, itr] = ratio
            itr = itr + 1
    global_scale_l = torch.median(bone_ratios[:, 0:-width_bone_num], dim=1)
    global_scale_w = torch.median(bone_ratios[:, -width_bone_num:], dim=1)
    global_scale = torch.mean(torch.cat([global_scale_l.values, global_scale_w.values]))
    return global_scale.view(-1, 1)

def ComputeGlobalR(target_skeleton_batch, mano_skeleton_batch, isBody=False, ismano=True):
    batch_size = target_skeleton_batch.shape[0]
    global_rot = np.zeros([batch_size, 3])

    for i in range(batch_size):
        mano_skeleton = mano_skeleton_batch[i, :, :]
        target_skeleton = target_skeleton_batch[i, :, :]
        # palm and index
        if isBody:
            smpl_spine = normalizev(mano_skeleton[1] - mano_skeleton[8])
            smpl_shoulder = normalizev(mano_skeleton[2] - mano_skeleton[5])
            smpl_facing = normalizev(np.cross(smpl_spine, smpl_shoulder))

            target_spine = normalizev(target_skeleton[1] - target_skeleton[8])
            target_shoulder = normalizev(target_skeleton[2] - target_skeleton[5])
            target_facing = normalizev(np.cross(target_spine, target_shoulder))

            # compute spine rot from mano to target
            axis = normalizev(np.cross(smpl_spine, target_spine))
            angle = np.arccos(np.dot(smpl_spine, target_spine))
            rot1 = R.from_rotvec(axis * angle)

            smpl_facing_rot1 = R.apply(rot1, smpl_facing)
            axis2 = normalizev(np.cross(smpl_facing_rot1, target_facing))
            angle2 = np.arccos(np.dot(smpl_facing_rot1, target_facing))
            rot2 = R.from_rotvec(axis2 * angle2)
            est_rot = rot2 * rot1

            # fix 180
            rot_mat = est_rot.as_dcm()
            trans_smpl_spine = np.matmul(rot_mat, smpl_spine)
            trans_smpl_shoulder = np.matmul(rot_mat, smpl_shoulder)
            trans_smpl_facing = np.matmul(rot_mat, smpl_facing)

            angle_facing = np.dot(trans_smpl_facing, target_facing)
            angle_spine = np.dot(trans_smpl_spine, target_spine)
            angle_shoulder = np.dot(trans_smpl_shoulder, target_shoulder)

            # rotate index again
            if angle_facing < 0:
                rot3 = R.from_rotvec(axis2 * 3.1415)
                est_rot = rot3 * est_rot
                rot_mat = est_rot.as_dcm()
                trans_smpl_facing = np.matmul(rot_mat, smpl_facing)
                angle_facing = np.dot(trans_smpl_facing, target_facing)

            global_rot[i, :] = est_rot.as_rotvec()

        else:
            if ismano:
                middle = 9
                pinky = 17
            else:
                middle = 11
                pinky = 21

            # right hand axis
            mano_index = mano_skeleton[middle] - mano_skeleton[0]
            mano_pinky = mano_skeleton[pinky] - mano_skeleton[0]
            mano_palm = np.cross(mano_index, mano_pinky)
            mano_index = normalizev(mano_index)
            mano_palm = normalizev(mano_palm)
            mano_cross = normalizev(np.cross(mano_palm, mano_index))
            mano_indexfix = normalizev(np.cross(mano_palm, mano_cross))

            targ_index = target_skeleton[middle] - target_skeleton[0]
            targ_pinky = target_skeleton[pinky] - target_skeleton[0]
            targ_palm = np.cross(targ_index, targ_pinky)
            targ_index = normalizev(targ_index)
            targ_palm = normalizev(targ_palm)
            targ_cross = normalizev(np.cross(targ_palm, targ_index))
            targ_indexfix = normalizev(np.cross(targ_palm, targ_cross))

            # compute palm rot from mano to target
            axis = normalizev(np.cross(mano_palm, targ_palm))
            angle = np.arccos(np.dot(mano_palm, targ_palm))
            rot1 = R.from_rotvec(axis * angle)
            est_rot = rot1

            mano_indexfix = R.apply(rot1, mano_indexfix)
            axis2 = normalizev(np.cross(mano_indexfix, targ_indexfix))
            angle2 = np.arccos(np.dot(mano_indexfix, targ_indexfix))
            rot2 = R.from_rotvec(axis2 * angle2)
            est_rot = rot2 * rot1

            while True:
                rot_mat = est_rot.as_dcm()
                trans_mano_index = np.matmul(rot_mat, mano_index)
                trans_mano_palm = np.matmul(rot_mat, mano_palm)
                trans_mano_cross = np.matmul(rot_mat, mano_cross)

                angle_index = np.dot(trans_mano_index, targ_index)
                angle_palm = np.dot(trans_mano_palm, targ_palm)
                angle_cross = np.dot(trans_mano_cross, targ_cross)

                if np.isclose(angle_index, 1.0) and np.isclose(angle_palm, 1.0) and np.isclose(angle_cross, 1.0):
                    break

                # rotate index again
                if angle_index < 0:
                    rot22 = R.from_rotvec(axis2 * np.pi)
                    est_rot = rot22 * est_rot

            global_rot[i, :] = est_rot.as_rotvec()

    return global_rot