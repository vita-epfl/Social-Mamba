import argparse
import json
import random
from dataclasses import dataclass

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


@dataclass
class Robot:
    position: np.ndarray
    goal: np.ndarray
    velocity: np.ndarray
    max_speed: float
    radius: float

    def update(self, force, dt):
        self.velocity = self.velocity + force * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / (speed + 1e-8) * self.max_speed
        self.position = self.position + self.velocity * dt


def inference(model, input_joints, padding_mask, out_len):
    model.eval()
    with torch.no_grad():
        pred_traj, _, _, _ = model(input_joints, padding_mask)
    return pred_traj[:, :, -out_len:].detach().cpu()  # (K, B, F, 2)


def unit(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def cv_forecast(in_joints_b, agent_idx, out_f):
    last = in_joints_b[-1, agent_idx, :2]
    prev = in_joints_b[-2, agent_idx, :2] if in_joints_b.shape[0] >= 2 else last
    vel = last - prev
    steps = np.arange(1, out_f + 1, dtype=np.float32)[:, None]
    return last[None] + steps * vel[None]


def zero_forecast(in_joints_b, agent_idx, out_f):
    last = in_joints_b[-1, agent_idx, :2]
    return np.repeat(last[None], out_f, axis=0)


def obstacle_list(method, in_joints_b, out_joints_b, valid_agents, model_pred_b, neighbor_policy, out_f):
    obstacles = []

    def add(traj, weight=1.0, source=""):
        obstacles.append({"traj": np.asarray(traj, dtype=np.float32), "weight": float(weight), "source": source})

    if method == "zero":
        for n in valid_agents:
            add(zero_forecast(in_joints_b, n, out_f), source="zero")
    elif method == "cv":
        for n in valid_agents:
            add(cv_forecast(in_joints_b, n, out_f), source="cv")
    elif method == "gt":
        for n in valid_agents:
            add(out_joints_b[:, n, :2], source="gt")
    elif method == "model":
        # The existing forecasting model predicts the focal/ego pedestrian. Neighbors are held
        # fixed across methods using the selected simple policy, so the comparison isolates the
        # downstream impact of the learned focal forecast.
        num_modes = model_pred_b.shape[0]
        for k in range(num_modes):
            add(model_pred_b[k], weight=1.0 / max(num_modes, 1), source="model_focal")
        for n in valid_agents:
            if n == 0:
                continue
            if neighbor_policy == "none":
                continue
            if neighbor_policy == "zero":
                add(zero_forecast(in_joints_b, n, out_f), source="neighbor_zero")
            elif neighbor_policy == "cv":
                add(cv_forecast(in_joints_b, n, out_f), source="neighbor_cv")
            elif neighbor_policy == "gt":
                add(out_joints_b[:, n, :2], source="neighbor_gt")
            else:
                raise ValueError(f"Unknown neighbor policy: {neighbor_policy}")
    else:
        raise ValueError(f"Unknown method: {method}")

    return obstacles


def compute_force(robot, obstacles, step, args):
    direction_to_goal = robot.goal - robot.position
    total_force = args.k_attractive * unit(direction_to_goal)

    for obstacle in obstacles:
        traj = obstacle["traj"]
        weight = obstacle["weight"]
        for tau in range(step, min(len(traj), step + args.lookahead)):
            obs_pos = traj[tau]
            if not np.all(np.isfinite(obs_pos)):
                continue
            direction = robot.position - obs_pos
            distance = np.linalg.norm(direction)
            if not np.isfinite(distance) or distance < 1e-6 or distance > args.influence_radius:
                continue
            temporal_weight = 1.0 / (1.0 + tau - step)
            clearance = max(distance - args.robot_radius - args.ped_radius, 1e-3)
            magnitude = args.k_repulsive * weight * temporal_weight / clearance
            total_force += magnitude * direction / distance

    return total_force


def simulate_robot(obstacles, out_f, args):
    robot = Robot(
        position=np.array([args.start_x, args.start_y], dtype=np.float32),
        goal=np.array([args.goal_x, args.goal_y], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        max_speed=args.max_speed,
        radius=args.robot_radius,
    )
    positions = []
    velocities = []
    for step in range(out_f):
        force = compute_force(robot, obstacles, step, args)
        robot.update(force, args.dt)
        positions.append(robot.position.copy())
        velocities.append(robot.velocity.copy())
    return np.asarray(positions), np.asarray(velocities)


def path_metrics(path, velocities, gt_agents, args):
    horizon = min(len(path), gt_agents.shape[1])
    if horizon == 0 or gt_agents.shape[0] == 0:
        min_dist = float("inf")
        collision = False
        near = False
    else:
        dists = np.linalg.norm(path[None, :horizon] - gt_agents[:, :horizon], axis=-1)
        finite_dists = dists[np.isfinite(dists)]
        if finite_dists.size == 0:
            min_dist = float("inf")
            collision = False
            near = False
        else:
            min_dist = float(np.min(finite_dists))
            collision = bool(np.any(finite_dists < args.robot_radius + args.ped_radius))
            near = bool(np.any(finite_dists < args.near_threshold))

    diffs = np.diff(path, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=-1) if len(path) > 1 else np.asarray([0.0])
    finite_steps = step_lengths[np.isfinite(step_lengths)]
    path_length = float(finite_steps.sum()) if finite_steps.size else 0.0
    accel = np.diff(velocities, axis=0)
    accel_norm = np.linalg.norm(accel, axis=-1) if len(accel) > 0 else np.asarray([0.0])
    finite_accel = accel_norm[np.isfinite(accel_norm)]
    smoothness = float(finite_accel.mean()) if finite_accel.size else 0.0
    final_goal_dist_raw = np.linalg.norm(path[-1] - np.array([args.goal_x, args.goal_y], dtype=np.float32))
    final_goal_dist = float(final_goal_dist_raw) if np.isfinite(final_goal_dist_raw) else float("inf")
    return {
        "collision": collision,
        "near_miss": near,
        "min_dist": min_dist,
        "success": final_goal_dist <= args.success_threshold,
        "path_length": path_length,
        "smoothness": smoothness,
        "final_goal_dist": final_goal_dist,
    }


def focal_forecast_errors(method, in_joints_b, out_joints_b, model_pred_b, out_f):
    gt = out_joints_b[:, 0, :2]
    if method == "zero":
        pred = zero_forecast(in_joints_b, 0, out_f)[None]
    elif method == "cv":
        pred = cv_forecast(in_joints_b, 0, out_f)[None]
    elif method == "gt":
        pred = gt[None]
    elif method == "model":
        pred = model_pred_b
    else:
        raise ValueError(method)
    dist = np.linalg.norm(pred - gt[None], axis=-1)
    ade = float(np.min(dist.mean(axis=-1)))
    fde = float(np.min(dist[:, -1]))
    return ade, fde


def evaluate(model, dataloader, config, methods, args):
    out_f = config["TRAIN"]["output_track_size"]
    stats = {m: {"n": 0, "collisions": 0, "near_misses": 0, "successes": 0, "min_dist": 0.0,
                 "path_length": 0.0, "smoothness": 0.0, "final_goal_dist": 0.0,
                 "forecast_ade": 0.0, "forecast_fde": 0.0} for m in methods}
    bar = Bar("PLANNING", fill="#", max=len(dataloader) if args.max_batches is None else min(len(dataloader), args.max_batches))
    seen = 0

    for batch_id, batch in enumerate(dataloader):
        if args.max_batches is not None and batch_id >= args.max_batches:
            break
        joints, masks, padding_mask = batch
        joints = joints[:, :, :, :1, :]
        masks = masks[:, :, :, :1]
        padding_mask = padding_mask.to(config["DEVICE"])
        in_joints, _, out_joints, _, padding_mask, _ = batch_process_coords_eval(joints, masks, padding_mask, config)

        model_pred = inference(model, in_joints, padding_mask, out_len=out_f) if "model" in methods else None
        in_np = in_joints.cpu().numpy()
        out_np = out_joints.cpu().numpy()
        pad_np = padding_mask.detach().cpu().numpy().astype(bool)

        batch_size = in_np.shape[0]
        for b in range(batch_size):
            if args.max_scenes is not None and seen >= args.max_scenes:
                break
            valid_agents = np.where(~pad_np[b])[0].tolist()
            if 0 not in valid_agents:
                continue
            gt_agents = out_np[b, :, valid_agents, :2].transpose(1, 0, 2)
            model_pred_b = model_pred[:, b, :, :2].numpy() if model_pred is not None else None

            for method in methods:
                obstacles = obstacle_list(method, in_np[b], out_np[b], valid_agents, model_pred_b, args.neighbor_policy, out_f)
                path, velocities = simulate_robot(obstacles, out_f, args)
                metrics = path_metrics(path, velocities, gt_agents, args)
                ade, fde = focal_forecast_errors(method, in_np[b], out_np[b], model_pred_b, out_f)
                s = stats[method]
                s["n"] += 1
                s["collisions"] += int(metrics["collision"])
                s["near_misses"] += int(metrics["near_miss"])
                s["successes"] += int(metrics["success"])
                s["min_dist"] += metrics["min_dist"]
                s["path_length"] += metrics["path_length"]
                s["smoothness"] += metrics["smoothness"]
                s["final_goal_dist"] += metrics["final_goal_dist"]
                s["forecast_ade"] += ade
                s["forecast_fde"] += fde
            seen += 1
        bar.next()
        if args.max_scenes is not None and seen >= args.max_scenes:
            break
    bar.finish()

    summary = {}
    for method, s in stats.items():
        n = max(s["n"], 1)
        summary[method] = {
            "samples": s["n"],
            "collision_rate": s["collisions"] / n,
            "near_miss_rate": s["near_misses"] / n,
            "success_rate": s["successes"] / n,
            "min_distance": s["min_dist"] / n,
            "path_length": s["path_length"] / n,
            "smoothness": s["smoothness"] / n,
            "final_goal_distance": s["final_goal_dist"] / n,
            "forecast_ADE": s["forecast_ade"] / n,
            "forecast_FDE": s["forecast_fde"] / n,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", default="zero,cv,model,gt")
    parser.add_argument("--neighbor-policy", choices=["none", "zero", "cv", "gt"], default="cv")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--robot-radius", type=float, default=0.5)
    parser.add_argument("--ped-radius", type=float, default=0.2)
    parser.add_argument("--near-threshold", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=-5.0)
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=5.0)
    parser.add_argument("--lookahead", type=int, default=12)
    parser.add_argument("--influence-radius", type=float, default=2.0)
    parser.add_argument("--k-repulsive", type=float, default=1.0)
    parser.add_argument("--k-attractive", type=float, default=1.0)
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

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summary = evaluate(model, dataloader, config, methods, args)

    print("\nPlanning summary")
    print("method\tforecast_ADE\tforecast_FDE\tcollision_rate\tnear_miss_rate\tsuccess_rate\tmin_dist\tpath_len\tsmoothness\tsamples")
    for method in methods:
        s = summary[method]
        print(f"{method}\t{s['forecast_ADE']:.6f}\t{s['forecast_FDE']:.6f}\t{s['collision_rate']:.6f}\t{s['near_miss_rate']:.6f}\t{s['success_rate']:.6f}\t{s['min_distance']:.6f}\t{s['path_length']:.6f}\t{s['smoothness']:.6f}\t{s['samples']}")
    print("JSON_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
