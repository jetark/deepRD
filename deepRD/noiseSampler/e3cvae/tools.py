import torch
from deepRD.noiseSampler.cvae.transforms import minimal_image_rel

def radial_embedding(edge_vec, num_basis=16, r_cut=5.0):
    "Embedding for the edge vector."
    r = edge_vec.norm(dim=-1, keepdim=True)
    centers = torch.linspace(0.0, r_cut, num_basis, device=edge_vec.device)
    widths = (r_cut / num_basis)
    return torch.exp(-((r - centers) ** 2) / (widths ** 2))

def pack_e3_features(scalars, vectors):
    """
    scalars: [N, n_scalar]
    vectors: [N, n_vector, 3]
    matching irreps: "{n_vector}x1o + {n_scalar}x0e"
    """
    return torch.cat([
        vectors.reshape(vectors.shape[0], -1),
        scalars,
    ], dim=-1)

def build_dimer_graph_batch(q1, q2, v1, v2, r1, r2, v1_prev, v2_prev, r1_prev, r2_prev, r1_next=None, r2_next=None):
    """
    Shapes:
        q1, q2, v1, ...: [B, 3]
    Returns graph tensors for B two-node graphs.
    """

    B = q1.shape[0]
    device = q1.device

    # Node-wise vectors: [B, 2, num_vecs, 3]
    vec_dec = torch.stack([
        torch.stack([v1, v2], dim=1),
        torch.stack([v1_prev, v2_prev], dim=1),
        torch.stack([r1, r2], dim=1),
        torch.stack([r1_prev, r2_prev], dim=1),
    ], dim=2)

    # flatten nodes: [B*2, num_vecs, 3]
    vec_dec = vec_dec.reshape(B * 2, 4, 3)

    # example scalar features
    scal_dec = torch.cat([
        vec_dec.norm(dim=-1),  # [B*2, 4]
    ], dim=-1)

    h_dec_base = pack_e3_features(scal_dec, vec_dec)

    if r1_next is not None:
        target = torch.stack([r1_next, r2_next], dim=1).reshape(B * 2, 3)

        vec_enc = torch.cat([
            vec_dec,
            target[:, None, :],
        ], dim=1)

        scal_enc = torch.cat([
            vec_enc.norm(dim=-1),
        ], dim=-1)

        h_enc = pack_e3_features(scal_enc, vec_enc)
    else:
        target = None
        h_enc = None

    # edges: for every graph, 0->1 and 1->0
    node_offset = 2 * torch.arange(B, device=device)

    src = torch.cat([node_offset + 0, node_offset + 1], dim=0)
    dst = torch.cat([node_offset + 1, node_offset + 0], dim=0)
    edge_index = torch.stack([src, dst], dim=0)

    x12 = minimal_image_rel(q1, q2)  # replace with your PBC function
    edge_vec = torch.cat([x12, -x12], dim=0)

    edge_radial = radial_embedding(edge_vec)

    batch_index = torch.arange(B, device=device).repeat_interleave(2)

    return {
        "h_dec_base": h_dec_base,
        "h_enc": h_enc,
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "edge_radial": edge_radial,
        "batch_index": batch_index,
        "r_next": target,
        "num_graphs": B,
    }

