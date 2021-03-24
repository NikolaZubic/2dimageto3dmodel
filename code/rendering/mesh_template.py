import kaolin as kal
import numpy as np
import torch
import torch.nn.functional as F
import math
import os

from .utils import grid_sample_bilinear, circpad

from packaging import version

class MeshTemplate:
    
    def __init__(self, mesh_path, is_symmetric=True):
        
        MeshTemplate._monkey_patch_dependencies()
        
        mesh = kal.rep.TriangleMesh.from_obj(mesh_path, enable_adjacency=True)
        mesh.cuda()

        poles = [mesh.vertices[:, 1].argmax().item(), mesh.vertices[:, 1].argmin().item()] # North pole, south pole

        # Compute reflection information (for mesh symmetry)
        axis = 0
        if version.parse(torch.__version__) < version.parse('1.2'):
            neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4)[:, 0].cpu().numpy()
            zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4)[:, 0].cpu().numpy()
        else:
            neg_indices = torch.nonzero(mesh.vertices[:, axis] < -1e-4, as_tuple=False)[:, 0].cpu().numpy()
            zero_indices = torch.nonzero(torch.abs(mesh.vertices[:, axis]) < 1e-4, as_tuple=False)[:, 0].cpu().numpy()
            
        pos_indices = []
        for idx in neg_indices:
            opposite_vtx = mesh.vertices[idx].clone()
            opposite_vtx[axis] *= -1
            dists = (mesh.vertices - opposite_vtx).norm(dim=-1)
            minval, minidx = torch.min(dists, dim=0)
            assert minval < 1e-4, minval
            pos_indices.append(minidx.item())
        assert len(pos_indices) == len(neg_indices)
        assert len(pos_indices) == len(set(pos_indices)) # No duplicates
        pos_indices = np.array(pos_indices)

        pos_indices = torch.LongTensor(pos_indices).cuda()
        neg_indices = torch.LongTensor(neg_indices).cuda()
        zero_indices = torch.LongTensor(zero_indices).cuda()
        nonneg_indices = torch.LongTensor(list(pos_indices) + list(zero_indices)).cuda()

        total_count = len(pos_indices) + len(neg_indices) + len(zero_indices)
        assert total_count == len(mesh.vertices), (total_count, len(mesh.vertices))

        index_list = {}
        segments = 32
        rings = 31 if '31rings' in mesh_path else 16


        for faces, vertices in zip(mesh.face_textures, mesh.faces):
            for face, vertex in zip(faces, vertices):
                if vertex.item() not in index_list:
                    index_list[vertex.item()] = []
                res = mesh.uvs[face].cpu().numpy() * [segments, rings]
                if math.isclose(res[0], segments, abs_tol=1e-4):
                    res[0] = 0 # Wrap around
                index_list[vertex.item()].append(res)

        topo_map = torch.zeros(mesh.vertices.shape[0], 2)
        for idx, data in index_list.items():
            avg = np.mean(np.array(data, dtype=np.float32), axis=0) / [segments, rings]
            topo_map[idx] = torch.Tensor(avg)

        # Flip topo map
        topo_map = topo_map * 2 - 1
        topo_map = topo_map * torch.FloatTensor([1, -1]).to(topo_map.device)
        topo_map = topo_map.cuda()
        nonneg_topo_map = topo_map[nonneg_indices]

        # Force x = 0 for zero-indices if symmetry is enabled
        symmetry_mask = torch.ones_like(mesh.vertices).unsqueeze(0)
        symmetry_mask[:, zero_indices, 0] = 0

        # Compute mesh tangent map (per-vertex normals, tangents, and bitangents)
        mesh_normals = F.normalize(mesh.vertices, dim=1)
        up_vector = torch.Tensor([[0, 1, 0]]).to(mesh_normals.device).expand_as(mesh_normals)
        mesh_tangents = F.normalize(torch.cross(mesh_normals, up_vector, dim=1), dim=1)
        mesh_bitangents = torch.cross(mesh_normals, mesh_tangents, dim=1)
        # North pole and south pole have no (bi)tangent
        mesh_tangents[poles[0]] = 0
        mesh_bitangents[poles[0]] = 0
        mesh_tangents[poles[1]] = 0
        mesh_bitangents[poles[1]] = 0
        
        tangent_map = torch.stack((mesh_normals, mesh_tangents, mesh_bitangents), dim=1).cuda()
        nonneg_tangent_map = tangent_map[nonneg_indices] # For symmetric meshes
        
        self.mesh = mesh
        self.topo_map = topo_map
        self.nonneg_topo_map = nonneg_topo_map
        self.nonneg_indices = nonneg_indices
        self.neg_indices = neg_indices
        self.pos_indices = pos_indices
        self.symmetry_mask = symmetry_mask
        self.tangent_map = tangent_map
        self.nonneg_tangent_map = nonneg_tangent_map
        self.is_symmetric = is_symmetric
        
    def deform(self, deltas):
        """
        Deform this mesh template along its tangent map, using the provided vertex displacements.
        """
        tgm = self.nonneg_tangent_map if self.is_symmetric else self.tangent_map
        return (deltas.unsqueeze(-2) @ tgm.expand(deltas.shape[0], -1, -1, -1)).squeeze(-2)

    def compute_normals(self, vertex_positions):
        """
        Compute face normals from the *final* vertex positions (not deltas).
        """
        a = vertex_positions[:, self.mesh.faces[:, 0]]
        b = vertex_positions[:, self.mesh.faces[:, 1]]
        c = vertex_positions[:, self.mesh.faces[:, 2]]
        v1 = b - a
        v2 = c - a
        normal = torch.cross(v1, v2, dim=2)
        return F.normalize(normal, dim=2)

    def get_vertex_positions(self, displacement_map):
        """
        Deform this mesh template using the provided UV displacement map.
        Output: 3D vertex positions in object space.
        """
        topo = self.nonneg_topo_map if self.is_symmetric else self.topo_map
        _, displacement_map_padded = self.adjust_uv_and_texture(displacement_map)
        if self.is_symmetric:
            # Compensate for even symmetry in UV map
            delta = 1/(2*displacement_map.shape[3])
            expansion = (displacement_map.shape[3]+1)/displacement_map.shape[3]
            topo = topo.clone()
            topo[:, 0] = (topo[:, 0] + 1 + 2*delta - expansion)/expansion # Only for x axis
        topo_expanded = topo.unsqueeze(0).unsqueeze(-2).expand(displacement_map.shape[0], -1, -1, -1)
        vertex_deltas_local = grid_sample_bilinear(displacement_map_padded, topo_expanded).squeeze(-1).permute(0, 2, 1)
        vertex_deltas = self.deform(vertex_deltas_local)
        if self.is_symmetric:
            # Symmetrize
            vtx_n = torch.Tensor(vertex_deltas.shape[0], self.topo_map.shape[0], 3).to(vertex_deltas.device)
            vtx_n[:, self.nonneg_indices] = vertex_deltas
            vtx_n2 = vtx_n.clone()
            vtx_n2[:, self.neg_indices] = vtx_n[:, self.pos_indices] * torch.Tensor([-1, 1, 1]).to(vtx_n.device)
            vertex_deltas = vtx_n2 * self.symmetry_mask
        vertex_positions = self.mesh.vertices.unsqueeze(0) + vertex_deltas
        return vertex_positions
    
    def adjust_uv_and_texture(self, texture, return_texture=True):
        """
        Returns the UV coordinates of this mesh template,
        and preprocesses the provided texture to account for boundary conditions.
        If the mesh is symmetric, the texture and UVs are adjusted accordingly.
        """
        
        if self.is_symmetric:
            delta = 1/(2*texture.shape[3])
            expansion = (texture.shape[3]+1)/texture.shape[3]
            uvs = self.mesh.uvs.clone()
            uvs[:, 0] = (uvs[:, 0] + delta)/expansion
            
            uvs = uvs.expand(texture.shape[0], -1, -1)
            texture = circpad(texture, 1) # Circular padding
        else:
            uvs = self.mesh.uvs.expand(texture.shape[0], -1, -1)
            texture = torch.cat((texture, texture[:, :, :, :1]), dim=3)
            
        return uvs, texture
    
    def forward_renderer(self, renderer, vertex_positions, texture, num_gpus=1, **kwargs):
        mesh_faces = self.mesh.faces
        mesh_face_textures = self.mesh.face_textures
        if num_gpus > 1:
            mesh_faces = mesh_faces.repeat(num_gpus, 1)
            mesh_face_textures = mesh_face_textures.repeat(num_gpus, 1)

        input_uvs, input_texture = self.adjust_uv_and_texture(texture)

        image, alpha, _ = renderer(points=[vertex_positions, mesh_faces],
                                   uv_bxpx2=input_uvs,
                                   texture_bx3xthxtw=input_texture,
                                   ft_fx3=mesh_face_textures,
                                   **kwargs)
        return image, alpha
    
    def export_obj(self, path_prefix, vertex_positions, texture):
        assert len(vertex_positions.shape) == 2
        mesh_path = path_prefix + '.obj'
        material_path = path_prefix + '.mtl'
        material_name = os.path.basename(path_prefix)
        
        # Export mesh .obj
        with open(mesh_path, 'w') as file:
            print('mtllib ' + os.path.basename(material_path), file=file)
            for v in vertex_positions:
                print('v {:.5f} {:.5f} {:.5f}'.format(*v), file=file)
            for uv in self.mesh.uvs:
                print('vt {:.5f} {:.5f}'.format(*uv), file=file)
            print('usemtl ' + material_name, file=file)
            for f, ft in zip(self.mesh.faces, self.mesh.face_textures):
                print('f {}/{} {}/{} {}/{}'.format(f[0]+1, ft[0]+1, f[1]+1, ft[1]+1, f[2]+1, ft[2]+1), file=file)
                
        # Export material .mtl
        with open(material_path, 'w') as file:
            print('newmtl ' + material_name, file=file)
            print('Ka 1.000 1.000 1.000', file=file)
            print('Kd 1.000 1.000 1.000', file=file)
            print('Ks 0.000 0.000 0.000', file=file)
            print('d 1.0', file=file)
            print('illum 1', file=file)
            print('map_Ka ' + material_name + '.png', file=file)
            print('map_Kd ' + material_name + '.png', file=file)
            
        # Export texture
        import imageio
        texture = (texture.permute(1, 2, 0)*255).clamp(0, 255).cpu().byte().numpy()
        imageio.imwrite(path_prefix + '.png', texture)
                
    @staticmethod
    def _monkey_patch_dependencies():
        if version.parse(torch.__version__) < version.parse('1.2'):
            def torch_where_patched(*args, **kwargs):
                if len(args) == 1:
                    return (torch.nonzero(args[0]), )
                else:
                    return torch._where_original(*args)

            torch._where_original = torch.where
            torch.where = torch_where_patched
            
        if version.parse(torch.__version__) >= version.parse('1.5'):
            from .monkey_patches import compute_adjacency_info_patched
            # Monkey patch
            kal.rep.Mesh.compute_adjacency_info = staticmethod(compute_adjacency_info_patched)
                
                