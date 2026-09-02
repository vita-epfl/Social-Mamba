import argparse
import random
import time

import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import DataLoader

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


def best_of_k_errors(pred_joints, out_joints):
    # pred_joints: (K, B, F, 2), out_joints: (B, F, N, C). Evaluate focal agent N=0.
    gt_xy = out_joints[:, :, 0, :2].to(pred_joints.device)
    dist = torch.linalg.norm(pred_joints[:, :, :, :2].permute(1, 0, 2, 3) - gt_xy[:, None], dim=-1)
    ade = dist.mean(dim=-1).min(dim=-1).values
    fde = dist[:, :, -1].min(dim=-1).values
    return ade.sum().item(), fde.sum().item(), gt_xy.shape[0]


def evaluate(model, dataloader, config, max_batches=None, warmup_batches=5):
    device = config["DEVICE"]
    out_f = config["TRAIN"]["output_track_size"]
    use_cuda = isinstance(device, str) and device.startswith("cuda")

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    total_time_ms = 0.0
    timed_batches = 0
    ade_sum = 0.0
    fde_sum = 0.0
    n_samples = 0
    max_items = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    bar = Bar("EFFICIENCY", fill="#", max=max_items)

    for batch_id, batch in enumerate(dataloader):
        if max_batches is not None and batch_id >= max_batches:
            break

        joints, masks, padding_mask = batch
        joints = joints[:, :, :, :1, :]
        masks = masks[:, :, :, :1]
        padding_mask = padding_mask.to(device)
        in_joints, _, out_joints, _, padding_mask, _ = batch_process_coords_eval(joints, masks, padding_mask, config)
        in_joints = in_joints.to(device)

        if use_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred_joints = inference(model, in_joints, padding_mask, out_f)
        if use_cuda:
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if batch_id >= warmup_batches:
            total_time_ms += elapsed_ms
            timed_batches += 1

        batch_ade, batch_fde, batch_n = best_of_k_errors(pred_joints.detach().cpu(), out_joints.cpu())
        ade_sum += batch_ade
        fde_sum += batch_fde
        n_samples += batch_n
        bar.next()

    bar.finish()
    avg_time_ms = total_time_ms / max(timed_batches, 1)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_cuda else 0.0
    return {
        "samples": n_samples,
        "batches_timed": timed_batches,
        "ADE": ade_sum / max(n_samples, 1),
        "FDE": fde_sum / max(n_samples, 1),
        "avg_batch_latency_ms": avg_time_ms,
        "peak_gpu_memory_mib": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to a Social-Mamba checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit the number of batches")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Batches excluded from latency average")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logger = create_logger("")
    ckpt = torch.load(args.ckpt, map_location=torch.device("cpu"))
    config = ckpt["config"]
    config["DEVICE"] = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    model = create_model_UMamba(config, logger, config["DEVICE"])
    state_dict = {key.replace("module.", ""): value for key, value in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(config["DEVICE"])

    in_f = config["TRAIN"]["input_track_size"]
    out_f = config["TRAIN"]["output_track_size"]
    dataset_name = config["DATA"]["train_datasets"][0]
    dataset = create_dataset(dataset_name, logger, split=args.split, track_size=in_f + out_f, track_cutoff=in_f)
    batch_size = args.batch_size or max(1, config["TRAIN"]["batch_size"] // config["DATA"].get("chunk_size", 4))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=config["TRAIN"].get("num_workers", 0), shuffle=False, collate_fn=collate_batch)

    summary = evaluate(model, dataloader, config, max_batches=args.max_batches, warmup_batches=args.warmup_batches)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
