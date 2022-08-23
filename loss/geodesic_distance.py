import torch
import numpy as np
import multiprocessing as mp
import gdist


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/geodesic.html#geodesic_distance

def geodesic_distance(pos, face, src=None, dest=None, norm=True,
                      max_distance=None, num_workers=0):
    r"""Computes (normalized) geodesic distances of a mesh given by :obj:`pos`
    and :obj:`face`. If :obj:`src` and :obj:`dest` are given, this method only
    computes the geodesic distances for the respective source and target
    node-pairs.

    .. note::

        This function requires the :obj:`gdist` package.
        To install, run :obj:`pip install cython && pip install gdist`.

    Args:
        pos (Tensor): The node positions.
        face (LongTensor): The face indices.
        src (LongTensor, optional): If given, only compute geodesic distances
            for the specified source indices. (default: :obj:`None`)
        dest (LongTensor, optional): If given, only compute geodesic distances
            for the specified target indices. (default: :obj:`None`)
        norm (bool, optional): Normalizes geodesic distances by
            :math:`\sqrt{\textrm{area}(\mathcal{M})}`. (default: :obj:`True`)
        max_distance (float, optional): If given, only yields results for
            geodesic distances less than :obj:`max_distance`. This will speed
            up runtime dramatically. (default: :obj:`None`)
        num_workers (int, optional): How many subprocesses to use for
            calculating geodesic distances.
            :obj:`0` means that computation takes place in the main process.
            :obj:`-1` means that the available amount of CPU cores is used.
            (default: :obj:`0`)

    :rtype: Tensor
    """

    max_distance = float('inf') if max_distance is None else max_distance

    if norm:
        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        norm = (area.norm(p=2, dim=1) / 2).sum().sqrt().item()
    else:
        norm = 1.0

    dtype = pos.dtype

    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().to(torch.double).numpy()
        face = face.detach().cpu().to(torch.int).numpy()
    else:
        pos = np.array(pos, dtype=np.double)
        face = np.array(face, dtype=np.int32)

    if src is None and dest is None:
        out = gdist.local_gdist_matrix(pos, face, max_distance * norm).toarray() / norm
        return out

    if src is None:
        src = np.arange(pos.shape[0], dtype=np.int32)
    elif isinstance(pos, torch.Tensor):
        src = src.detach().cpu().to(torch.int).numpy()
    else:
        src = np.array(src, dtype=np.int32)
    
    if dest is None:
        dest = None
    elif isinstance(dest, torch.Tensor):
        dest = dest.detach().cpu().to(torch.int).numpy()
    else:
        dest = np.array(dest, dtype=np.int32)

    def _parallel_loop(pos, face, src, dest, max_distance, norm, i, dtype):
        s = src[i:i + 1]
        d = None if dest is None else dest[i:i + 1]
        out = gdist.compute_gdist(pos, face, s, d, max_distance * norm) / norm
        return out

    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            outs = pool.starmap(
                _parallel_loop,
                [(pos, face, src, dest, max_distance, norm, i, dtype)
                 for i in range(len(src))])
    else:
        outs = [
            _parallel_loop(pos, face, src, dest, max_distance, norm, i, dtype)
            for i in range(len(src))
        ]

    out = torch.cat(outs, dim=0)

    if dest is None:
        out = out.view(-1, pos.shape[0])

    return out
