import torch
import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes
from loss.contactloss import self_collision_loss
from loss.loss_collecter import ARAP_reg_loss, collision_loss, p3d_chamfer_distance_with_filter, update_p3d_mesh_shape_prior_losses
from model.tet_fem import NHObject
from loss.geodesic_distance import geodesic_distance

class TetNonRigidNet():
    def __init__(self, template_mesh, target_mesh, tet_template_dict, weight_dict, device, _compute_align=True, _compute_collision=True, save_tmp=False, save_folder=None, args=None) -> None:
        if args is not None:
            self.args = args

        self._semantic_large_reg = args.semantic_large_reg

        self._tet_template_dict = tet_template_dict
        
        self._compute_align = _compute_align
        self._compute_collision = _compute_collision

        self.device = device

        self.save_tmp = save_tmp
        self.save_folder = save_folder

        # init data
        self.template_v = template_mesh.verts_packed()
        self.template_f = template_mesh.faces_packed()
        self.target_mesh = target_mesh

        # define forward
        self.init_nh_obj()

        self.tet_node_offset = torch.zeros_like(tet_template_dict['tet_v_rest']).requires_grad_(True)
        self.iter_counter_tet = torch.tensor(-1, requires_grad=False).to(device)
        self.tet_node_reg_weight = self.load_tet_node_reg_weight()
        
        # loss dict
        self.weight_dict = weight_dict
        
        self.forward_tet()

    def load_tet_node_reg_weight(self):
        if "reg_mask" in self._tet_template_dict:
            reg_mask = self._tet_template_dict["reg_mask"]
            reg_weight = torch.ones_like(reg_mask)
            reg_mask = reg_mask.bool()
            reg_weight[reg_mask] *= self._semantic_large_reg
        else:
            reg_weight = torch.ones(self.tet_node_offset.shape[0]).to(self.device)

        return reg_weight

    def init_nh_obj(self):
        rest_v = self._tet_template_dict['tet_v_rest']
        rest_f = self._tet_template_dict['tet_f']
        rest_e = self._tet_template_dict['tet_e']

        if "tet_v_rest_gd" in self._tet_template_dict:
            self.tet_v_rest_gd =self._tet_template_dict['tet_v_rest_gd']
        else:
            out_dist = geodesic_distance(rest_v, rest_f, norm=False, num_workers=-1)
            self.tet_v_rest_gd = out_dist

        self.tet_v_rest = rest_v

        # inner verts
        v_rest_surface_mask = torch.zeros(rest_v.shape[0])
        v_rest_surface_mask[torch.unique(rest_f.reshape(-1))] = 1
        v_rest_surface_mask = v_rest_surface_mask.clone().detach().bool()
        self.tet_v_rest_gd[~v_rest_surface_mask, :] = 1e5
        self.tet_v_rest_gd[:, ~v_rest_surface_mask] = 1e5

        flip_rest_f = self.template_f

        # initianlized with template
        self.nh_obj_net = NHObject(rest_v, flip_rest_f, rest_e, v_rest=rest_v, E=self.args.nh_E, poisson_v=self.args.nh_p)

        self.tet_v_init = self.template_v 
        self.nh_obj_net.update(self.tet_v_init)
        

    def forward_tet(self):
        self.iter_counter_tet += 1
        
        deformed_tet_v, deformed_tet_fn, tet_nh_loss = self.nh_obj_net.next_step(self.tet_v_init, self.tet_node_offset)
        
        self.deformed_tet_v = deformed_tet_v
        self.deformed_tet_fn = deformed_tet_fn

        tet_mean_nh_loss = tet_nh_loss.mean()
        tet_max_nh_loss = tet_nh_loss.max()

        return tet_mean_nh_loss, tet_max_nh_loss


    def get_mesh(self):
        deformed_template_meshes = {}
        deformed_template_meshes["tet"] = Meshes([self.deformed_tet_v[:,:3].detach()], [self.nh_obj_net.face])
        return deformed_template_meshes

        

    def sample_points_on_tetmesh(self):
        tetmesh = Meshes(verts=self.deformed_tet_v[:,:3].unsqueeze(0), faces=self.nh_obj_net.face.unsqueeze(0))
        sample_v, sample_v_normal = pytorch3d.ops.sample_points_from_meshes(tetmesh, self.args.simplify_mesh_v, return_normals=True)
        sample_v_n = torch.cat([sample_v, sample_v_normal], dim=-1)
        return sample_v_n.squeeze()

    def optim_tet(self):
        tet_mean_nh_loss, tet_max_nh_loss = self.forward_tet()
        
        loss_dict = {}
        # physical loss
        if self.weight_dict['nh'] > 0:
            loss_dict['nh'] = tet_mean_nh_loss * self.weight_dict['nh']
        if self.weight_dict['nh_max'] > 0:
            loss_dict['nh_max'] = tet_max_nh_loss * self.weight_dict['nh_max']
        
        update_p3d_mesh_shape_prior_losses(self.deformed_tet_v, self.nh_obj_net.face, loss_dict, self.weight_dict)

        if self._compute_align:
            if self.weight_dict['smooth'] > 0:
                reg_loss, reg_mask = ARAP_reg_loss(self.tet_v_init[:,:3], None, self.tet_node_offset, node_gd=self.tet_v_rest_gd, thres_corres=self.args.THRES_CORRES, node_sigma=self.args.smooth_sigma)
                reg_loss *= self.tet_node_reg_weight[reg_mask]
                loss_dict['smooth'] = reg_loss.mean() * self.weight_dict['smooth']

            if self.weight_dict['data'] > 0:
                sample_trg_v, sample_trg_n = sample_points_from_meshes(self.target_mesh, self.args.simplify_mesh_v, return_normals=True)
                sample_trg_v = sample_trg_v.squeeze()
                sample_trg_n = sample_trg_n.squeeze()
                sample_src = self.sample_points_on_tetmesh()

                cf_dis, cf_dis_normal  = p3d_chamfer_distance_with_filter(
                    sample_src[:,:3].unsqueeze(0),
                    sample_trg_v.unsqueeze(0),
                    x_normals=sample_src[:,3:].unsqueeze(0),
                    y_normals=sample_trg_n.unsqueeze(0),
                    angle_filter=self.args.angle_filter,
                    distance_filter=self.args.distance_filter,  wx=1, wy=0)
                # raw_cf_dis and raw_cf_normal (1-cos)
                
                loss_dict['data'] = cf_dis * self.weight_dict['data']
                
                if self.weight_dict['data_normal'] > 0:
                    loss_dict['data_normal'] = cf_dis_normal * self.weight_dict['data_normal']


        if self._compute_collision:
            # if self.weight_dict['collision'] > 0:  # for muscles
            #     loss_dict['collision'] = collision_loss(self._muscle_name, self.deformed_tet_v, self.computed_muscles, self.target_piano, self.target_piano_f, self.weight_dict, 
            #     lab=self.args.contact_lab, 
            #     distance_mode=self.args.contact_distance_mode, 
            #     use_mean=self.args.contact_distance_use_mean, contact_list=self.contact_list, 
            #     contact_thres=self.args.contact_thres)
                
            if self.weight_dict['self_collision'] > 0:
                loss_dict['self_collision'] = self_collision_loss(self.deformed_tet_v[:,:3], self.deformed_tet_v[:,3:],
                self.nh_obj_net.face, self.deformed_tet_fn, self.tet_v_rest.squeeze(), self.nh_obj_net.element.squeeze(), self.weight_dict, 
                distance_mode=self.args.contact_distance_mode)



        return loss_dict
