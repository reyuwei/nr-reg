import torch
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance

def p3d_point_mesh_face_distance(points, meshes, dual=False, close_thres=16):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl)` distance
    """
    pcls = Pointclouds([points[:, :3]])

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, fids = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    if dual:
        # mesh to point distance
        face_to_point = face_point_distance(
            points, points_first_idx, tris, tris_first_idx, max_tris
        )
        count_in = face_to_point < close_thres
        face_to_point_count_in = face_to_point[count_in]
        
        dis = point_to_face.mean() + face_to_point_count_in.mean()
        return dis
    else:
        return point_to_face, fids

def p3d_point_meshvf_face_distance(points, mesh_v, mesh_f):
    meshes = Meshes(verts=[mesh_v[:, :3]], faces=[mesh_f])
    return p3d_point_mesh_face_distance(points, meshes)


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

# Full batch mode
def batch_mesh_contains_points(
    ray_origins,
    obj_triangles,
    direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]),
    ray_directions=None, # (batch, point_nb, 3)
    return_hit=False, # return hit point 
    face_normal=None, #(batch_size, triangle_nb, 3)
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh

    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    device = ray_origins.device
    direction = direction.to(device)
    tol_thresh = 1e-5
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    if ray_directions is not None:
        batch_direction = ray_directions
        pvec = torch.cross(batch_direction.unsqueeze(1).repeat(1, triangle_nb, 1, 1), 
                           v0v2.unsqueeze(2).repeat(1, 1, point_nb, 1))
        dets = torch.einsum("btc, btvc->btv", v0v1, pvec)
        parallel = abs(dets) < tol_thresh
        invdet = 1 / (dets + 0.1 * tol_thresh)
        pvec = reshape_fortran(pvec, [batch_size, triangle_nb*point_nb, pvec.shape[-1]])
        invdet = reshape_fortran(invdet, [invdet.shape[0], -1])
        parallel = reshape_fortran(parallel, [parallel.shape[0], -1])
        batch_direction = batch_direction.unsqueeze(1).repeat(1, triangle_nb, 1, 1)
        batch_direction = reshape_fortran(batch_direction, [batch_direction.shape[0], -1, batch_direction.shape[-1]])
    else:
        # Expand needed vectors
        batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)
        # Compute ray/triangle intersections
        pvec = torch.cross(batch_direction, v0v2, dim=2)
        dets = torch.bmm(
            v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
        ).view(batch_size, triangle_nb)
        # Check if ray and triangle are parallel
        parallel = abs(dets) < tol_thresh
        invdet = 1 / (dets + 0.1 * tol_thresh)
        pvec = pvec.repeat(1, point_nb, 1)
        invdet = invdet.repeat(1, point_nb)
        parallel = parallel.repeat(1, point_nb)
        batch_direction = batch_direction.repeat(1, point_nb, 1)




    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    # # Check that all intersection conditions are met
    not_parallel = parallel.logical_not()
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0

    if return_hit:
        hit_t = t.view(batch_size, point_nb, triangle_nb)
        ignore_direction_mask = (v_correct * u_correct * not_parallel).view(batch_size, point_nb, triangle_nb)
        hit_t[~ignore_direction_mask] = 1e10
        hit_t_per_v, hit_t_per_v_f = torch.min(hit_t.abs(), dim=-1)
        fakedim = torch.arange(0, point_nb, dtype=torch.long).to(device)
        hit_t_per_v_with_sign = hit_t[:, fakedim, hit_t_per_v_f].view(hit_t_per_v.shape)
        closest_hit_point = ray_origins + ray_directions * hit_t_per_v_with_sign.unsqueeze(-1)

        # compute hit point normal
        closest_hit_point_normal = face_normal.reshape(-1, 3)[hit_t_per_v_f.reshape(-1)].reshape(face_normal.shape[0], -1, 3)

        return exterior, closest_hit_point, closest_hit_point_normal
    else:
        return exterior


def batch_det33(b, c, d):
    # b, c, d: (batch_size, N, 3) 
    mat33 = torch.stack([b, c, d], dim=-1)
    det = torch.det(mat33)
    return det

def batch_point_in_tet_barycoordinate(p, obj_tets):
    # https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not
    device = p.device
    batch_size = obj_tets.shape[0]
    triangle_nb = obj_tets.shape[1]
    point_nb = p.shape[1]

    batch_points_size = batch_size * triangle_nb
    v0, v1, v2, v3 = obj_tets[:, :, 0], obj_tets[:, :, 1], obj_tets[:, :, 2], obj_tets[:, :, 3]

    v0 = v0.repeat(1, point_nb, 1)
    v1 = v1.repeat(1, point_nb, 1)
    v2 = v2.repeat(1, point_nb, 1)
    v3 = v3.repeat(1, point_nb, 1)

    p_expand = p.view(batch_size, point_nb, 1, 3).repeat(1, 1, triangle_nb, 1)
    p_expand = p_expand.reshape(batch_size, triangle_nb*point_nb, 3)

    a = v0 - p_expand
    b = v1 - p_expand
    c = v2 - p_expand
    d = v3 - p_expand

    detA = batch_det33(b,c,d)
    detB = batch_det33(a,c,d)
    detC = batch_det33(a,b,d)
    detD = batch_det33(a,b,c)
    ret0 = (detA > 0.0) & (detB < 0.0) & (detC > 0.0) & (detD < 0.0)
    ret1 = (detA < 0.0) & (detB > 0.0) & (detC < 0.0) & (detD > 0.0)
    
    intet = (ret0 | ret1).view(batch_size, point_nb, triangle_nb)
    tet_id = torch.nonzero(intet, as_tuple=False)
    tet_id = tet_id.view(-1, 3) # [point_nb, 3] # one vertex could be in two tets
    # find barycentric coordinate
    bcoo = batch_barycentric_tet(obj_tets[:, tet_id[:, -1]], p[:, tet_id[:, -2]])

    return bcoo, tet_id[:, -1], tet_id[:, -2]

def ScTP(a, b, c):
    return (a * torch.cross(b, c)).sum(-1)

def batch_barycentric_tet(tet_v, p):
    # https://stackoverflow.com/questions/38545520/barycentric-coordinates-of-a-tetrahedron
    a, b, c, d = tet_v[:,:, 0], tet_v[:,:, 1], tet_v[:,:, 2], tet_v[:,:, 3]
    vap = p-a
    vbp = p-b
    
    vab = b-a
    vac = c-a
    vad = d-a

    vbc = c-b
    vbd = d-b

    va6 = ScTP(vbp, vbd, vbc)
    vb6 = ScTP(vap, vac, vad)
    vc6 = ScTP(vap, vad, vab)
    vd6 = ScTP(vap, vab, vac)
    v6 = 1 / ScTP(vab, vac, vad)

    coo = torch.stack([va6*v6, vb6*v6, vc6*v6, vd6*v6], dim=-1)

    # test
    # tt = (coo.unsqueeze(-1) * tet_v).sum(-2)
    # print(tt - p).sum()

    return coo


