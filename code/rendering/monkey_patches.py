# At the time of writing, Kaolin is not compatible with PyTorch >= 1.5,
# so we monkey patch the methods we need to make it work with newer versions.

# Adapted from: https://github.com/NVIDIAGameWorks/kaolin/blob/e7e513173bd4159ae45be6b3e156a3ad156a3eb9/kaolin/rep/Mesh.py#L586-L735

import torch

def compute_adjacency_info_patched(vertices: torch.Tensor, faces: torch.Tensor):
    """Build data structures to help speed up connectivity queries. Assumes
    a homogeneous mesh, i.e., each face has the same number of vertices.

    The outputs have the following format: AA, AA_count

    AA_count: ``[count_0, ..., count_n]``

    with AA:

    .. code-block::

        [[aa_{0,0}, ..., aa_{0,count_0} (, -1, ..., -1)],
        [aa_{1,0}, ..., aa_{1,count_1} (, -1, ..., -1)],
                    ...
        [aa_{n,0}, ..., aa_{n,count_n} (, -1, ..., -1)]]
    """

    device = vertices.device
    facesize = faces.shape[1]
    nb_vertices = vertices.shape[0]
    nb_faces = faces.shape[0]
    edges = torch.cat([faces[:,i:i+2] for i in range(facesize - 1)] +
                      [faces[:,[-1,0]]], dim=0)
    # Sort the vertex of edges in increasing order
    edges = torch.sort(edges, dim=1)[0]
    # id of corresponding face in edges
    face_ids = torch.arange(nb_faces, device=device, dtype=torch.long).repeat(facesize)
    # remove multiple occurences and sort by the first vertex
    # the edge key / id is fixed from now as the first axis position
    # edges_ids will give the key of the edges on the original vector
    edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
    nb_edges = edges.shape[0]

    # EDGE2EDGES
    _edges_ids = edges_ids.reshape(facesize, nb_faces)
    edges2edges = torch.cat([
        torch.stack([_edges_ids[1:], _edges_ids[:-1]], dim=-1).reshape(-1, 2),
        torch.stack([_edges_ids[-1:], _edges_ids[:1]], dim=-1).reshape(-1, 2)
    ], dim=0)

    double_edges2edges = torch.cat([edges2edges, torch.flip(edges2edges, dims=(1,))], dim=0)
    double_edges2edges = torch.cat(
        [double_edges2edges, torch.arange(double_edges2edges.shape[0], device=device, dtype=torch.long).reshape(-1, 1)], dim=1)
    double_edges2edges = torch.unique(double_edges2edges, sorted=True, dim=0)[:,:2]
    idx_first = torch.where(
        torch.nn.functional.pad(double_edges2edges[1:,0] != double_edges2edges[:-1,0],
                                (1, 0), value=1))[0]
    nb_edges_per_edge = idx_first[1:] - idx_first[:-1]
    offsets = torch.zeros(double_edges2edges.shape[0], device=device, dtype=torch.long)
    offsets[idx_first[1:]] = nb_edges_per_edge
    sub_idx = (torch.arange(double_edges2edges.shape[0], device=device,dtype=torch.long) -
               torch.cumsum(offsets, dim=0))
    nb_edges_per_edge = torch.cat([nb_edges_per_edge,
                                   double_edges2edges.shape[0] - idx_first[-1:]],
                                  dim=0)
    max_sub_idx = torch.max(nb_edges_per_edge)
    ee = torch.full((nb_edges, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
    ee[double_edges2edges[:,0], sub_idx] = double_edges2edges[:,1]

    # EDGE2FACE
    sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
    sorted_faces_ids = face_ids[order_edges_ids]
    # indices of first occurences of each key
    idx_first = torch.where(
        torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                                (1,0), value=1))[0]
    nb_faces_per_edge = idx_first[1:] - idx_first[:-1]
    # compute sub_idx (2nd axis indices to store the faces)
    offsets = torch.zeros(sorted_edges_ids.shape[0], device=device, dtype=torch.long)
    offsets[idx_first[1:]] = nb_faces_per_edge
    sub_idx = (torch.arange(sorted_edges_ids.shape[0], device=device, dtype=torch.long) -
               torch.cumsum(offsets, dim=0))
    # TODO(cfujitsang): potential way to compute sub_idx differently
    #                   to test with bigger model
    #sub_idx = torch.ones(sorted_edges_ids.shape[0], device=device, dtype=torch.long)
    #sub_idx[0] = 0
    #sub_idx[idx_first[1:]] = 1 - nb_faces_per_edge
    #sub_idx = torch.cumsum(sub_idx, dim=0)
    nb_faces_per_edge = torch.cat([nb_faces_per_edge,
                                   sorted_edges_ids.shape[0] - idx_first[-1:]],
                                  dim=0)
    max_sub_idx = torch.max(nb_faces_per_edge)
    ef = torch.full((nb_edges, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
    ef[sorted_edges_ids, sub_idx] = sorted_faces_ids
    # FACE2FACES
    nb_faces_per_face = torch.stack([nb_faces_per_edge[edges_ids[i*nb_faces:(i+1)*nb_faces]]
                                     for i in range(facesize)], dim=1).sum(dim=1) - facesize
    ff = torch.cat([ef[edges_ids[i*nb_faces:(i+1)*nb_faces]] for i in range(facesize)], dim=1)
    # remove self occurences
    ff[ff == torch.arange(nb_faces, device=device, dtype=torch.long).view(-1,1)] = -1
    ff = torch.sort(ff, dim=-1, descending=True)[0]
    to_del = (ff[:,1:] == ff[:,:-1]) & (ff[:,1:] != -1)
    ff[:,1:][to_del] = -1
    nb_faces_per_face = nb_faces_per_face - torch.sum(to_del, dim=1)
    max_sub_idx = torch.max(nb_faces_per_face)
    ff = torch.sort(ff, dim=-1, descending=True)[0][:,:max_sub_idx]

    # VERTEX2VERTICES and VERTEX2EDGES
    npy_edges = edges.cpu().numpy()
    edge2key = {tuple(npy_edges[i]): i for i in range(nb_edges)}
    #_edges and double_edges 2nd axis correspond to the triplet:
    # [left vertex, right vertex, edge key]
    _edges = torch.cat([edges, torch.arange(nb_edges, device=device).view(-1, 1)],
                       dim=1)
    double_edges = torch.cat([_edges, _edges[:,[1,0,2]]], dim=0)
    double_edges = torch.unique(double_edges, sorted=True, dim=0)
    # TODO(cfujitsang): potential improvment, to test with bigger model:
    #double_edges0, order_double_edges = torch.sort(double_edges[0])
    nb_double_edges = double_edges.shape[0]
    # indices of first occurences of each key
    idx_first = torch.where(
        torch.nn.functional.pad(double_edges[1:,0] != double_edges[:-1,0],
                                (1,0), value=1))[0]
    nb_edges_per_vertex = idx_first[1:] - idx_first[:-1]
    # compute sub_idx (2nd axis indices to store the edges)
    offsets = torch.zeros(nb_double_edges, device=device, dtype=torch.long)
    offsets[idx_first[1:]] = nb_edges_per_vertex
    sub_idx = (torch.arange(nb_double_edges, device=device, dtype=torch.long) -
               torch.cumsum(offsets, dim=0))
    nb_edges_per_vertex = torch.cat([nb_edges_per_vertex,
                                     nb_double_edges - idx_first[-1:]], dim=0)
    max_sub_idx = torch.max(nb_edges_per_vertex)
    vv = torch.full((nb_vertices, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
    vv[double_edges[:,0], sub_idx] = double_edges[:,1]
    ve = torch.full((nb_vertices, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
    ve[double_edges[:,0], sub_idx] = double_edges[:,2]

    # VERTEX2FACES
    vertex_ordered, order_vertex = torch.sort(faces.view(-1))
    face_ids_in_vertex_order = order_vertex // facesize # This line has been patched
    # indices of first occurences of each id
    idx_first = torch.where(
        torch.nn.functional.pad(vertex_ordered[1:] != vertex_ordered[:-1], (1,0), value=1))[0]
    nb_faces_per_vertex = idx_first[1:] - idx_first[:-1]
    # compute sub_idx (2nd axis indices to store the faces)
    offsets = torch.zeros(vertex_ordered.shape[0], device=device, dtype=torch.long)
    offsets[idx_first[1:]] = nb_faces_per_vertex
    sub_idx = (torch.arange(vertex_ordered.shape[0], device=device, dtype=torch.long) -
               torch.cumsum(offsets, dim=0))
    # TODO(cfujitsang): it seems that nb_faces_per_vertex == nb_edges_per_vertex ?
    nb_faces_per_vertex = torch.cat([nb_faces_per_vertex,
                                     vertex_ordered.shape[0] - idx_first[-1:]], dim=0)
    max_sub_idx = torch.max(nb_faces_per_vertex)
    vf = torch.full((nb_vertices, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
    vf[vertex_ordered, sub_idx] = face_ids_in_vertex_order

    return edge2key, edges, vv, nb_edges_per_vertex, ve, nb_edges_per_vertex, vf, \
        nb_faces_per_vertex, ff, nb_faces_per_face, ee, nb_edges_per_edge, ef, nb_faces_per_edge