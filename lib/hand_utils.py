import numpy as np

MUSCLE_NAMES = ["m5", "m45", "m34", "m23", "m13", "m1","m12"]
SURFACE_NAMES = ["skin_3mm"]

# CARPAL_JOINTS = [0, 1,2,3, 5,6,7, 10,11,12, 15,16,17, 20,21,22]
ROOT_JOINT_IDX = 0

CARPAL_JOINTS = range(25)

CARPAL_JOINTS_REG = [4, 8, 12, 16]

DOF2_BONES = [1, 2, 4, 5, 8, 9, 12, 13, 16, 17]
DOF1_BONES = [3, 6, 7, 10, 11, 14, 15, 18, 19]

JOINT_TO_BONE = [1,2,3, 5,6,7,8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23]

JOINT_ID_NAME_DICT = {
    0: "carpal",
    1: "met1",
    2: "pro1",
    3: "dis1",
    4: "dis1_end",

    5: "met2",
    6: "pro2",
    7: "int2",
    8: "dis2",
    9: "dis2_end",

    10: "met3",
    11: "pro3",
    12: "int3",
    13: "dis3",
    14: "dis3_end",

    15: "met4",
    16: "pro4",
    17: "int4",
    18: "dis4",
    19: "dis4_end",

    20: "met5",
    21: "pro5",
    22: "int5",
    23: "dis5",
    24: "dis5_end"
}

JOINT_NAME_ID_DICT = {
    "carpal": 0,
    "met1": 1,
    "pro1": 2,
    "dis1": 3,
    "dis1_end": 4,

    "met2": 5,
    "pro2": 6,
    "int2": 7,
    "dis2": 8,
    "dis2_end": 9,

    "met3": 10,
    "pro3": 11,
    "int3": 12,
    "dis3": 13,
    "dis3_end": 14,

    "met4": 15,
    "pro4": 16,
    "int4": 17,
    "dis4": 18,
    "dis4_end": 19,

    "met5": 20,
    "pro5": 21,
    "int5": 22,
    "dis5": 23,
    "dis5_end": 24
}


BONE_TO_JOINT_NAME = {
    0: "carpal",

    1: "met1",
    2: "pro1",
    3: "dis1",

    4: "met2",
    5: "pro2",
    6: "int2",
    7: "dis2",

    8: "met3",
    9: "pro3",
    10: "int3",
    11: "dis3",

    12: "met4",
    13: "pro4",
    14: "int4",
    15: "dis4",

    16: "met5",
    17: "pro5",
    18: "int5",
    19: "dis5",
}
JOINT_PARENT_ID_DICT = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,

    5: 0,
    6: 5,
    7: 6,
    8: 7,
    9: 8,

    10: 0,
    11: 10,
    12: 11,
    13: 12,
    14: 13,

    15: 0,
    16: 15,
    17: 16,
    18: 17,
    19: 18,

    20: 0,
    21: 20,
    22: 21,
    23: 22,
    24: 23
}

JOINT_CHILD_ID_DICT = {
    0: [1, 5, 10, 15, 20],
    1: [2],
    2: [3],
    3: [4],
    4: [],

    5: [6],
    6: [7],
    7: [8],
    8: [9],
    9: [],

    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: [],

    15: [16],
    16: [17],
    17: [18],
    18: [19],
    19: [],

    20: [21],
    21: [22],
    22: [23],
    23: [24],
    24: []
}

STATIC_JOINT_NUM = len(JOINT_ID_NAME_DICT.keys()) #25
STATIC_BONE_NUM = len(BONE_TO_JOINT_NAME.keys()) #20

JOINT_ID_BONE_DICT = {}
JOINT_ID_BONE = np.zeros(STATIC_BONE_NUM)
BONE_ID_JOINT_DICT = {}
for key in JOINT_ID_NAME_DICT:
    value = JOINT_ID_NAME_DICT[key]
    for key_b in BONE_TO_JOINT_NAME:
        if BONE_TO_JOINT_NAME[key_b] == value:
            JOINT_ID_BONE_DICT[key] = key_b
            BONE_ID_JOINT_DICT[key_b] = key
            JOINT_ID_BONE[key_b] = key


PARENT_PIANO = list(JOINT_PARENT_ID_DICT.values())


def visual_weights(bone_obj, bone_pts_weights):
    import trimesh
    import colorsys


    '''
    visulize_weight 
    
    Parameters:
    bone_obj: trimesh with N vertices / np.array [N, 3]
    bone_pts_weights: np.array [N,k]

    Return:
    colored bone obj (trimesh.mesh or trimesh.pointcloud)
    '''

    label_list = np.argmax(bone_pts_weights, axis=1)
    bone_num = bone_pts_weights.shape[-1]
    if isinstance(bone_obj, np.ndarray):
        rgb = np.stack([colorsys.hsv_to_rgb(i * 1.0 / bone_num, 0.8, 0.8) for i in label_list])
        a = np.ones([rgb.shape[0], 1])
        rgba = np.hstack([rgb, a])
        bone_pts = trimesh.PointCloud(vertices=bone_obj.reshape(-1, 3), colors=rgba)
        return bone_pts
    else:
        if isinstance(bone_obj.visual, trimesh.visual.TextureVisuals):
            bone_obj.visual = bone_obj.visual.to_color()
            bone_obj.visual.vertex_colors = np.zeros([len(bone_obj.vertices), 4])
        for i in range(len(label_list)):
            (r, g, b) = colorsys.hsv_to_rgb(label_list[i] * 1.0 / bone_num, 0.8, 0.8)
            bone_obj.visual.vertex_colors[i] = (r * 255, g * 255, b * 255, 255)
        return bone_obj

def rbf_weights(volume, control_pts, control_pts_weights=None):
    """
    Compute 3D volume weight according to control points with rbf and "thin_plate" as kernel
    if control_pts_weights is None, directly return control points influences

    Parameters
    ----------
    volume : (np.array, [N,3]) volume points with unknown weights 
    control_pts : (np.array, [M,3]) control points
    control_pts_weights : (np.array, [M,K]) control point weights

    Returns
    ----------
    volume_weights : np.array, [N,K]
    """
    from scipy.interpolate import Rbf

    if control_pts_weights is None:
        control_pts_weights = np.eye(control_pts.shape[0])

    xyz = volume.reshape(-1, 3)
    chunk = 50000
    rbfi = Rbf(control_pts[:, 0], control_pts[:, 1], control_pts[:, 2], control_pts_weights, function="thin_plate", mode="N-D")
    # rbfi = Rbf(pts[:, 0], pts[:, 1], pts[:, 2], weight, function="multiquadric", mode="N-D")
    weight_volume = np.concatenate([rbfi(xyz[j:j + chunk, 0], xyz[j:j + chunk, 1], xyz[j:j + chunk, 2]) for j in
                                    range(0, xyz.shape[0], chunk)], 0)
    weight_volume[weight_volume < 0] = 0
    weight_volume = weight_volume / np.sum(weight_volume, axis=1).reshape(-1, 1)
    weight_volume = weight_volume.reshape(xyz.shape[0], -1)
    return weight_volume