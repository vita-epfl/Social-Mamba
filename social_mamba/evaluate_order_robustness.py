import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from progress.bar import Bar

try:
    from .dataset import create_dataset, collate_batch, batch_process_coords_eval
except ImportError:
    from dataset import create_dataset, collate_batch, batch_process_coords_eval
try:
    from .model import create_model_UMamba
except ImportError:
    from model import create_model_UMamba
try:
    from .utils.utils import create_logger
except ImportError:
    from utils.utils import create_logger


def inference(model, input_joints, padding_mask, out_len):
    model.eval()
    with torch.no_grad():
        pred_traj, _, _, _ = model(input_joints, padding_mask)
    return pred_traj[:, :, -out_len:]


def gather_agents(x, indices):
    if x is None:
        return None
    if x.dim() == 4:
        idx = indices[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3])
    elif x.dim() == 3:
        idx = indices[:, None, :].expand(-1, x.shape[1], -1)
    elif x.dim() == 2:
        idx = indices
    else:
        raise ValueError(f"Unsupported tensor rank {x.dim()}")
    return torch.gather(x, 2 if x.dim() in (3, 4) else 1, idx.to(x.device))


def permutation_indices(in_joints, padding_mask, kind, seed=0):
    batch_size, in_f, num_agents, _ = in_joints.shape
    device = in_joints.device
    if num_agents <= 1 or kind == "original":
        return torch.arange(num_agents, device=device).unsqueeze(0).repeat(batch_size, 1)

    neighbor_count = num_agents - 1
    if kind == "reverse":
        neigh = torch.arange(num_agents - 1, 0, -1, device=device).unsqueeze(0).repeat(batch_size, 1)
    elif kind == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        perms = [torch.randperm(neighbor_count, generator=gen, device=device) + 1 for _ in range(batch_size)]
        neigh = torch.stack(perms, dim=0)
    elif kind in {"distance", "risk"}:
        last_pos = in_joints[:, -1, :, :2]
        ego_pos = last_pos[:, 0:1]
        neighbor_pos = last_pos[:, 1:]
        rel_pos = neighbor_pos - ego_pos
        invalid_neighbors = padding_mask[:, 1:].to(device).bool()

        if kind == "distance":
            scores = torch.sum(rel_pos ** 2, dim=-1)
        else:
            prev_pos = in_joints[:, -2, :, :2]
            vel = last_pos - prev_pos
            rel_vel = vel[:, 1:] - vel[:, 0:1]
            closing = -torch.sum(rel_pos * rel_vel, dim=-1)
            dist_sq = torch.sum(rel_pos ** 2, dim=-1) + 1e-6
            scores = torch.where(closing > 0, dist_sq / (closing + 1e-6), torch.full_like(closing, float("inf")))
        scores = scores.masked_fill(invalid_neighbors, float("inf"))
        neigh = torch.argsort(scores, dim=1) + 1
    else:
        raise ValueError(f"Unknown permutation kind: {kind}")

    ego = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    return torch.cat([ego, neigh.long()], dim=1)


def apply_permutation(in_joints, in_masks, out_joints, out_masks, padding_mask, kind, seed=0):
    idx = permutation_indices(in_joints, padding_mask, kind, seed=seed)
    return (
        gather_agents(in_joints, idx),
        gather_agents(in_masks, idx),
        gather_agents(out_joints, idx),
        gather_agents(out_masks, idx),
        gather_agents(padding_mask, idx),
    )


def batch_min_ade_fde(pred_joints, out_joints):
    pred_xy = pred_joints.cpu().permute(1, 0, 2, 3)[:, :, :, :2]  # (B, K, F, 2)
    gt_xy = out_joints.cpu()[:, :, 0, :2]
    dist = torch.linalg.norm(pred_xy - gt_xy[:, None], dim=-1)
    ade = dist.mean(dim=-1).min(dim=1).values
    fde = dist[:, :, -1].min(dim=1).values
    return ade, fde, pred_xy


def prediction_shift(pred_xy, base_xy):
    # Mode labels are exchangeable, so compare each predicted mode to its closest original mode.
    # pred/base: (B, K, F, 2)
    pair = torch.linalg.norm(pred_xy[:, :, None] - base_xy[:, None], dim=-1).mean(dim=-1)
    return pair.min(dim=2).values.mean(dim=1)


def evaluate(model, dataloader, config, variants, random_seeds, max_batches=None):
    out_f = config["TRAIN"]["output_track_size"]
    stats = {name: {"ade_sum": 0.0, "fde_sum": 0.0, "shift_sum": 0.0, "n": 0} for name in variants}
    bar = Bar("ORDER ROBUSTNESS", fill="#", max=len(dataloader) if max_batches is None else min(len(dataloader), max_batches))

    for batch_id, batch in enumerate(dataloader):
        if max_batches is not None and batch_id >= max_batches:
            break
        joints, masks, padding_mask = batch
        joints = joints[:, :, :, :1, :]
        masks = masks[:, :, :, :1]
        padding_mask = padding_mask.to(config["DEVICE"])
        in_joints, in_masks, out_joints, out_masks, padding_mask, _ = batch_process_coords_eval(joints, masks, padding_mask, config)

        base_pred = inference(model, in_joints, padding_mask, out_len=out_f)
        base_ade, base_fde, base_xy = batch_min_ade_fde(base_pred, out_joints)
        stats["original"]["ade_sum"] += base_ade.sum().item()
        stats["original"]["fde_sum"] += base_fde.sum().item()
        stats["original"]["n"] += len(base_ade)

        for name in variants:
            if name == "original":
                continue
            if name.startswith("random"):
                seed = random_seeds[int(name.split("_")[1])]
                kind = "random"
            else:
                seed = 0
                kind = name
            p_in, _, p_out, _, p_padding = apply_permutation(in_joints, in_masks, out_joints, out_masks, padding_mask, kind, seed)
            pred = inference(model, p_in, p_padding, out_len=out_f)
            ade, fde, pred_xy = batch_min_ade_fde(pred, p_out)
            shift = prediction_shift(pred_xy, base_xy)
            stats[name]["ade_sum"] += ade.sum().item()
            stats[name]["fde_sum"] += fde.sum().item()
            stats[name]["shift_sum"] += shift.sum().item()
            stats[name]["n"] += len(ade)
        bar.next()
    bar.finish()

    summary = {}
    for name, item in stats.items():
        n = max(item["n"], 1)
        summary[name] = {
            "ADE": item["ade_sum"] / n,
            "FDE": item["fde_sum"] / n,
            "prediction_shift": item["shift_sum"] / n,
            "samples": item["n"],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--include-risk", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logger = create_logger("")
    ckpt = torch.load(args.ckpt, map_location=torch.device("cpu"))
    config = ckpt["config"]
    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"

    model = create_model_UMamba(config, logger, config["DEVICE"])
    pretrained_dict = {key.replace("module.", ""): value for key, value in ckpt["model"].items()}
    model.load_state_dict(pretrained_dict)

    dataset_name = config["DATA"]["train_datasets"][0]
    in_f = config["TRAIN"]["input_track_size"]
    out_f = config["TRAIN"]["output_track_size"]
    dataset = create_dataset(dataset_name, logger, split=args.split, track_size=in_f + out_f, track_cutoff=in_f)
    bs = max(1, config["TRAIN"]["batch_size"] // config["DATA"].get("chunk_size", 4))
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config["TRAIN"].get("num_workers", 0), shuffle=False, collate_fn=collate_batch)

    random_seeds = list(range(args.num_random))
    variants = ["original", "reverse"] + [f"random_{i}" for i in range(args.num_random)]
    if args.include_distance:
        variants.append("distance")
    if args.include_risk:
        variants.append("risk")

    summary = evaluate(model, dataloader, config, variants, random_seeds, max_batches=args.max_batches)
    original_ade = summary["original"]["ADE"]
    original_fde = summary["original"]["FDE"]
    for item in summary.values():
        item["delta_ADE"] = item["ADE"] - original_ade
        item["delta_FDE"] = item["FDE"] - original_fde

    print("\nOrder robustness summary")
    print("variant\tADE\tFDE\tdADE\tdFDE\tpred_shift\tsamples")
    for name in variants:
        s = summary[name]
        print(f"{name}\t{s['ADE']:.6f}\t{s['FDE']:.6f}\t{s['delta_ADE']:.6f}\t{s['delta_FDE']:.6f}\t{s['prediction_shift']:.6f}\t{s['samples']}")
    print("JSON_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
