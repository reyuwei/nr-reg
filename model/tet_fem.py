import torch
import numpy as np
from pytorch3d.structures.meshes import Meshes

class NHObject: # torch.tensor
    def __init__(self, v, f, e, B=None, V=None, v_rest=None, E=300, poisson_v=0.455):
    
        self.t = 0
        self.dt = 5e-4

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float()
            f = torch.from_numpy(f)
            e = torch.from_numpy(e)
        
        if v_rest is not None:
            if isinstance(v_rest, np.ndarray):
                v_rest = torch.from_numpy(v_rest).float()
        
        if B is not None:
            if isinstance(B, np.ndarray):
                B = torch.from_numpy(B).float()

        if V is not None:
            if isinstance(V, np.ndarray):
                V = torch.from_numpy(V).float()

        self.device = v.device
        self.vn = v.shape[0]
        self.fn = f.shape[0]
        self.en = e.shape[0]
        self.dim = 3

        ## for simulation
        self.E = E #300 # https://www.docin.com/p-1228898985.html
        self.poisson_v = poisson_v # 0.455 # Stable Neo-Hookean Flesh Simulation
        self.mu = self.E / (2*(1+self.poisson_v))
        self.la = (self.E * self.poisson_v) / ((1+self.poisson_v)*(1-2*self.poisson_v))

        # self.mu = 100 # lame's first parameter
        # self.la = 1000 # lame'2 second parameter / shear modulus

        # print("v: ", self.poisson_v, "mu:", self.mu, "la: ", self.la)
        print("vertices: ", self.vn, "elements: ", self.en)

        self.node = v
        self.face = f
        self.element = e
        self.surface_vid = torch.unique(self.face.reshape(-1), sorted=True)

        if B is None or V is None:
            self.node_rest = v_rest
            self.initialize()
        else:
            self.B = B
            self.element_volume_rest = V


    def initialize(self):
        self.B = torch.zeros([self.en, self.dim, self.dim]).to(self.device)
        self.element_volume_rest = torch.zeros(self.en).to(self.device)
        D = self.D(rest=True)
        self.B = torch.inverse(D) # rest
        self.element_volume_rest = torch.abs(torch.det(D)) / 6.0

    def update(self, v_lbs, force_init=False):
        if force_init:
            self.node_rest = v_lbs
            self.initialize()
        self.node = v_lbs

    def D(self, rest=False, new_node=None):
        a = self.element[:, 0]
        b = self.element[:, 1]
        c = self.element[:, 2]
        d = self.element[:, 3]

        if new_node is not None:
            v0 = (new_node[b]-new_node[a]).reshape(self.en, -1, 1)
            v1 = (new_node[c]-new_node[a]).reshape(self.en, -1, 1)
            v2 = (new_node[d]-new_node[a]).reshape(self.en, -1, 1)
        else:
            if rest:
                v0 = (self.node_rest[b]-self.node_rest[a]).reshape(self.en, -1, 1)
                v1 = (self.node_rest[c]-self.node_rest[a]).reshape(self.en, -1, 1)
                v2 = (self.node_rest[d]-self.node_rest[a]).reshape(self.en, -1, 1)
            else:
                v0 = (self.node[b]-self.node[a]).reshape(self.en, -1, 1)
                v1 = (self.node[c]-self.node[a]).reshape(self.en, -1, 1)
                v2 = (self.node[d]-self.node[a]).reshape(self.en, -1, 1)

        dmat = torch.cat([v0, v1, v2], dim=-1)
        return dmat

    def Psi(self, new_node=None): # (strain) energy density
        D_curr = self.D(rest=False, new_node=new_node)
        F = torch.bmm(D_curr, self.B)
        J = torch.det(F)
        # J[J < 0.01] = 0.01
        J_fix = torch.clamp(J, min=0.01)
        I_ = F @ torch.transpose(F, 1, 2)
        I = torch.einsum('bii->b', I_)
        psi = (self.mu / 2.0) * (I - self.dim) - self.mu * torch.log(J_fix) + (self.la / 2.0) * torch.log(J_fix)**2
        return psi

    def compute_normal(self, new_node):
        new_muscle_tet_mesh = Meshes([new_node], [self.face])
        new_muscle_tet_mesh._compute_vertex_normals()
        # flipped_deformed_muscle_normals = new_muscle_tet_mesh.verts_normals_packed() * -1.
        return new_muscle_tet_mesh.verts_normals_packed(), new_muscle_tet_mesh.faces_normals_packed()

    def next_step(self, nodes_rest, nodes_offset):
        self.t += 1
        
        new_node = nodes_rest + nodes_offset
        node_normal, face_normal = self.compute_normal(new_node)
        # self.node += nodes_offset

        # update U
        psi = self.Psi(new_node)
        eng = self.element_volume_rest * psi

        return torch.cat([new_node, node_normal], dim=1), face_normal, eng



def readtetfile(filename, index_start=0):
    # read nodes from *.node file

    v, f, e = [], [], []

    with open(filename+".node", "r") as file:
        vn = int(file.readline().split()[0])
        for i in range(vn):
            v.append([ float(x) for x in file.readline().split()[1:4]]) #[x , y, z]

    # read faces from *.face file (only for rendering)
    with open(filename+".face", "r") as file:
        fn = int(file.readline().split()[0])
        for i in range(fn):
            f.append([ int(ind)-index_start for ind in file.readline().split()[1:4] ]) # triangle

    # read elements from *.ele file
    with open(filename+".ele", "r") as file:
        en = int(file.readline().split()[0])
        for i in range(en): # !!!!! some files are 1-based indexed
            e.append([ int(ind)-index_start for ind in file.readline().split()[1:5] ]) # tetrahedron

    v = np.stack(v).reshape(-1, 3)
    f = np.stack(f).reshape(-1, 3).astype(np.int64)
    e = np.stack(e).reshape(-1, 4).astype(np.int64)

    save_dict = {
        "tet_v_rest": v,
        "tet_f": f,
        "tet_e": e,
    }

    return save_dict