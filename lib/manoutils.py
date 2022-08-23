import numpy as np
import torch
import chumpy as ch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

mano_select = torch.Tensor([
    0, 1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24]).type(torch.long)


class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
             for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel()
                           for pp in p]).ravel()

def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s), ))

def ready_arguments(fname_or_dict, posekey4vposed='pose'):
    import numpy as np
    import pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # dd = pickle.load(open(fname_or_dict, 'rb'))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in [
            'v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs',
            'betas', 'J'
    ]:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert (posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(pose_map_res)
    else:
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd_add = dd['posedirs'].dot(pose_map_res)
        dd['v_posed'] = dd['v_template'] + dd_add

    return dd


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def batch_rotprojs(batches_rotmats):
    proj_rotmats = []
    for batch_idx, batch_rotmats in enumerate(batches_rotmats):
        proj_batch_rotmats = []
        for rot_idx, rotmat in enumerate(batch_rotmats):
            # GPU implementation of svd is VERY slow
            # ~ 2 10^-3 per hit vs 5 10^-5 on cpu
            U, S, V = rotmat.cpu().svd()
            rotmat = torch.matmul(U, V.transpose(0, 1))
            orth_det = rotmat.det()
            # Remove reflection
            if orth_det < 0:
                rotmat[:, 2] = -1 * rotmat[:, 2]

            rotmat = rotmat.cuda()
            proj_batch_rotmats.append(rotmat)
        proj_rotmats.append(torch.stack(proj_batch_rotmats))
    return torch.stack(proj_rotmats)


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