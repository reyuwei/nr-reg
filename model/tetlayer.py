# from pytorch3d.loss import point_mesh_distance
# from pytorch3d.loss.chamfer import chamfer_distance
import os
import torch
import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes
from loss.contactloss import self_collision_loss
from loss.loss_collecter import ARAP_reg_loss, bone_attach_loss, collision_loss, p3d_chamfer_distance_with_filter, update_p3d_mesh_shape_prior_losses, link_loss
from model.tet_fem import NHObject

class TetNonRigidNet():
    def __init__(self, template_mesh, target_mesh, muscle_name, tet_template_dict, weight_dict, attach_dict, handside, device, attach_bone=None, _compute_align=True, _compute_collision=True, computed_muscles={}, contact_list={}, save_tmp=False, save_folder=None, tet_only=True, args=None
    ) -> None:
        if args is not None:
            self.args = args

        self._semantic_large_reg = args.semantic_large_reg

        self.tet_only = tet_only
        self._tet_template_dict = tet_template_dict
        self._attach_dict = attach_dict
        if muscle_name == "skin":
            self._muscle_name = "skin_3mm"
        else:
            self._muscle_name = "muscle"
        
        self._compute_align = _compute_align
        self._compute_collision = _compute_collision
        self._handside = handside

        self.device = device
        self.contact_list = contact_list
        self.computed_muscles = computed_muscles

        self.save_tmp = save_tmp
        self.save_folder = save_folder

        # init data
        self.template_muscle_v = template_mesh.verts_packed()
        self.template_muscle_f = template_mesh.faces_packed()
        self.target_muscle_mesh = target_mesh
        self.target_piano = attach_bone.verts_packed()
        self.target_piano_f = attach_bone.faces_packed()

        # define forward
        self.init_nh_obj()
       

        self.tet_node_offset = torch.zeros_like(tet_template_dict['tet_v_rest'][self._muscle_name]).requires_grad_(True)
        self.iter_counter_tet = torch.tensor(-1, requires_grad=False).to(device)
        self.tet_node_reg_weight = self.load_tet_node_reg_weight()
        self.tet_node_visible_face = self.load_tet_node_visible_face()
        
        # loss dict
        self.weight_dict = weight_dict
        
        self.forward_tet()

    def load_tet_node_reg_weight(self):
        if "reg_mask" in self._tet_template_dict:
            reg_mask = self._tet_template_dict["reg_mask"][self._muscle_name]
            reg_weight = torch.ones_like(reg_mask)
            reg_mask = reg_mask.bool()
            reg_weight[reg_mask] *= self._semantic_large_reg
        else:
            reg_weight = torch.ones(self.tet_node_offset.shape[0]).to(self.device)

        return reg_weight

    def load_tet_node_visible_face(self):
        if "new_face" in self._tet_template_dict:
            new_face = self._tet_template_dict['new_face']
            return new_face
        else:
            return None

    def init_nh_obj(self):
        rest_v = self._tet_template_dict['tet_v_rest'][self._muscle_name]
        rest_f = self._tet_template_dict['tet_f'][self._muscle_name]
        rest_e = self._tet_template_dict['tet_e'][self._muscle_name]
        self.tet_v_init_gd = self._tet_template_dict['tet_v_rest_gd'][self._muscle_name]
        self.tet_v_rest = rest_v

        # inner verts
        v_rest_surface_mask = torch.zeros(rest_v.shape[0])
        v_rest_surface_mask[torch.unique(rest_f.reshape(-1))] = 1
        v_rest_surface_mask = v_rest_surface_mask.clone().detach().bool()
        self.tet_v_init_gd[~v_rest_surface_mask, :] = 1e5
        self.tet_v_init_gd[:, ~v_rest_surface_mask] = 1e5

        flip_rest_f = self.template_muscle_f
        if self._handside == "left":
            rest_v[:, 1] *= -1

        # initianlized with template
        self.nh_obj_net = NHObject(rest_v, flip_rest_f, rest_e, v_rest=rest_v, E=self.args.nh_E, poisson_v=self.args.nh_p)

        if hasattr(self.args, 'NIMBLE_model'):
            print("init nh with nimble model")
            if "skin" in self._muscle_name:
                self.tet_v_rest = pytorch3d.io.load_objs_as_meshes([os.path.join(self.args.NIMBLE_model, "0_mean_tpi_skin.obj")]).to(self.device).verts_packed()
            else:
                self.tet_v_rest = pytorch3d.io.load_objs_as_meshes([os.path.join(self.args.NIMBLE_model, "0_mean_tpi_muscle.obj")]).to(self.device).verts_packed()
            if self._handside == "left":
                self.tet_v_rest[:,1] *= -1
            self.nh_obj_net.update(self.tet_v_rest, force_init=True)

        self.tet_v_init = self.template_muscle_v # self.batch['lbs_muscles_tet_nodes']
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

        deformed_muscle_meshes = {}

        try:
            if self._muscle_name in self._attach_dict[0]:
                attach_id = self._attach_dict[0][self._muscle_name]
                attach_tp = self.deformed_muscle_v[attach_id[:, 0], :3].detach()
                attach_tg = self.target_piano[attach_id[:,1], :3]
                save_deform_muscle = torch.cat([self.deformed_muscle_v[:, :3].detach(), attach_tp[:, :3], attach_tg[:, :3]], dim=0)
            else:
                save_deform_muscle = self.deformed_muscle_v[:,:3].detach()
            deformed_muscle_meshes[self._muscle_name+"_after"] = Meshes([save_deform_muscle], [self.muscle_f])
        except:
            pass
            # print("no embed mesh found")


        deformed_muscle_meshes[self._muscle_name+"_tet"] = Meshes([self.deformed_tet_v[:,:3].detach()], [self.nh_obj_net.face])

        return deformed_muscle_meshes

        

    def sample_points_on_tetmesh(self):
        if self.tet_node_visible_face is not None:
            tetmesh = Meshes(verts=self.deformed_tet_v[:,:3].unsqueeze(0), faces=self.tet_node_visible_face.unsqueeze(0))
        else:
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

        if not self.tet_only:
            # link loss
            if self.weight_dict['link'] > 0:
                loss_dict['link'] = link_loss(self._tet_template_dict, self._muscle_name, self.deformed_tet_v, self.deformed_muscle_v.detach(), self.weight_dict)
        else:

            if self._muscle_name in self._attach_dict[0]:
                if self.weight_dict['attach'] > 0:
                    attach_id = self._attach_dict[0][self._muscle_name]
                    loss_dict['attach'] = bone_attach_loss( self.deformed_tet_v, self.target_piano, attach_id, self.weight_dict)
            
            update_p3d_mesh_shape_prior_losses(self.deformed_tet_v, self.nh_obj_net.face, loss_dict, self.weight_dict)

            if self._compute_align:
                if self.weight_dict['smooth'] > 0:
                    reg_loss, reg_mask = ARAP_reg_loss(self.tet_v_init[:,:3], None, self.tet_node_offset, node_gd=self.tet_v_init_gd, thres_corres=self.args.THRES_CORRES, node_sigma=self.args.smooth_sigma)
                    reg_loss *= self.tet_node_reg_weight[reg_mask]
                    loss_dict['smooth'] = reg_loss.mean() * self.weight_dict['smooth']

                if self.weight_dict['data'] > 0:
                    sample_trg_v, sample_trg_n = sample_points_from_meshes(self.target_muscle_mesh, self.args.simplify_mesh_v, return_normals=True)
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

                if self.weight_dict['data_continue'] > 0 and hasattr(self, 'continue_tet_nodes'):
                    loss_dict['data_continue'] = ((self.continue_tet_nodes - self.deformed_tet_v[:,:3])**2).sum(-1).mean() * self.weight_dict['data_continue']


            if self._compute_collision:
                if self.weight_dict['collision'] > 0:
                    loss_dict['collision'] = collision_loss(self._muscle_name, self.deformed_tet_v, self.computed_muscles, self.target_piano, self.target_piano_f, self.weight_dict, 
                    lab=self.args.contact_lab, 
                    distance_mode=self.args.contact_distance_mode, 
                    use_mean=self.args.contact_distance_use_mean, contact_list=self.contact_list, 
                    contact_thres=self.args.contact_thres)
                    
                    # sample_src = self.sample_points_on_tetmesh()
                    # loss_dict['collision'] = collision_loss(self._muscle_name, sample_src, self.computed_muscles, self.target_piano, self.target_piano_f, self.weight_dict, 
                    # lab=self.args.contact_lab, 
                    # distance_mode=self.args.contact_distance_mode, 
                    # use_mean=self.args.contact_distance_use_mean, contact_list=self.contact_list, 
                    # contact_thres=self.args.contact_thres)
            
                if self.weight_dict['self_collision'] > 0:
                    loss_dict['self_collision'] = self_collision_loss(self.deformed_tet_v[:,:3], self.deformed_tet_v[:,3:],
                    self.nh_obj_net.face, self.deformed_tet_fn, self.tet_v_rest.squeeze(), self.nh_obj_net.element.squeeze(), self.weight_dict, 
                    distance_mode=self.args.contact_distance_mode)



        return loss_dict