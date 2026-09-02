import torch

def sort_social_grid(
    in_joints,
    in_masks,
    out_joints,
    out_masks,
    padding_mask,
    strategy='distance'
):
    """
    Sorts the social grid (agents) based on a specified strategy, keeping the
    ego agent (index 0) fixed.

    Args:
        in_joints (torch.Tensor): (B, in_F, N, 4)
        in_masks (torch.Tensor): (B, in_F, N)
        out_joints (torch.Tensor): (B, out_F, N, 4)
        out_masks (torch.Tensor): (B, out_F, N)
        padding_mask (torch.Tensor): (B, N). True/1 denotes padded agents.
        strategy (str): 'distance', 'random', 'kinematic', 'risk', 'intent', or 'none'.

    Returns:
        Tuple of permuted tensors:
        (in_joints, in_masks, out_joints, out_masks, padding_mask)
    """

    if strategy == 'none':
        return in_joints, in_masks, out_joints, out_masks, padding_mask

    batch_size, in_F, num_players, _ = in_joints.shape
    _, out_F, _, _ = out_joints.shape
    device = in_joints.device

    if num_players <= 1:
        return in_joints, in_masks, out_joints, out_masks, padding_mask

    # --- 1. Get Scores for Neighbors (index 1 to N-1) ---

    # (B, N-1)
    neighbor_padding_mask = padding_mask[:, 1:]

    if strategy == 'random':
        scores = torch.rand(batch_size, num_players - 1, device=device)

    else:
        last_pos = in_joints[:, -1, :, 0:2]
        ego_pos = last_pos[:, 0:1, :]
        neighbor_pos = last_pos[:, 1:, :]
        rel_pos = ego_pos - neighbor_pos

        if strategy == 'distance':
            scores = torch.sum(rel_pos**2, dim=-1) # (B, N-1)

        elif strategy in ['kinematic', 'risk', 'intent']:
            if in_F < 2:
                raise ValueError(
                    f"Strategy '{strategy}' requires at least 2 observation frames to compute velocity."
                )

            pos_t0 = in_joints[:, -2, :, 0:2]
            pos_t1 = in_joints[:, -1, :, 0:2]

            vel = pos_t1 - pos_t0
            ego_vel = vel[:, 0:1, :]
            neighbor_vel = vel[:, 1:, :]
            rel_vel = ego_vel - neighbor_vel

            if strategy == 'kinematic':
                rel_speed_sq = torch.sum(rel_vel**2, dim=-1) # (B, N-1)
                scores = -rel_speed_sq

                # --- FIX 1 (WAS ~neighbor_padding_mask) ---
                # We want invalid agents to be last (large positive score)
                scores[neighbor_padding_mask.bool()] = float('inf')
                # -------------------------------------------

            elif strategy == 'risk':
                dist_sq = torch.sum(rel_pos**2, dim=-1) + 1e-6
                proj = torch.sum(rel_pos * rel_vel, dim=-1)

                ttc = torch.full_like(proj, float('inf'))
                approaching_mask = proj > 0
                ttc[approaching_mask] = dist_sq[approaching_mask] / proj[approaching_mask]

                scores = ttc

            elif strategy == 'intent':
                vec_to_ego = rel_pos
                neighbor_vel_norm = neighbor_vel / (torch.norm(neighbor_vel, dim=-1, keepdim=True) + 1e-6)
                vec_to_ego_norm = vec_to_ego / (torch.norm(vec_to_ego, dim=-1, keepdim=True) + 1e-6)

                cosine_sim = torch.sum(neighbor_vel_norm * vec_to_ego_norm, dim=-1)
                scores = -cosine_sim
        else:
            raise ValueError(f"Unknown sorting strategy: {strategy}")

    # --- 2. Apply Masking to Scores ---
    # We set the score for all padded/invalid neighbors to infinity.
    if strategy != 'kinematic': # 'kinematic' already handled this

        # --- FIX 2 (WAS ~neighbor_padding_mask) ---
        scores[neighbor_padding_mask.bool()] = float('inf')
        # -------------------------------------------

    # --- 3. Get Sorted Indices ---
    sorted_neighbor_indices = torch.argsort(scores, dim=1)

    # --- 4. Create Full Index for Gathering ---
    sorted_neighbor_indices_abs = sorted_neighbor_indices + 1
    ego_idx = torch.zeros(batch_size, 1, device=device).long()
    full_sorted_indices = torch.cat([ego_idx, sorted_neighbor_indices_abs], dim=1)


    # --- 5. Permute All Tensors ---
    idx_in_j = full_sorted_indices.unsqueeze(1).unsqueeze(3).expand(-1, in_F, -1, 4)
    in_joints = torch.gather(in_joints, 2, idx_in_j)

    idx_out_j = full_sorted_indices.unsqueeze(1).unsqueeze(3).expand(-1, out_F, -1, 4)
    out_joints = torch.gather(out_joints, 2, idx_out_j)

    idx_in_m = full_sorted_indices.unsqueeze(1).expand(-1, in_F, -1)
    in_masks = in_masks.to(device)
    in_masks = torch.gather(in_masks, 2, idx_in_m)

    idx_out_m = full_sorted_indices.unsqueeze(1).expand(-1, out_F, -1)
    out_masks = out_masks.to(device)
    out_masks = torch.gather(out_masks, 2, idx_out_m)

    padding_mask = torch.gather(padding_mask, 1, full_sorted_indices)

    return in_joints, in_masks, out_joints, out_masks, padding_mask
