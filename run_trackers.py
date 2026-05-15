"""
run_trackers.py
---------------
Runs StrongSORT and ByteTrack on VIP-HTD dataset and evaluates
IDF1, IDSW, and MOTA.

VIP-HTD structure:
    VIP-HTD/
        clips/                          ← original videos (not used)
        mot-challenge-format/
            test/
                CAR_VS_BOS_001/
                    gt/gt.txt
                    img1/000001.jpg ...
                CAR_VS_NYR_001/
                ...
            train/
                ...
            validation/
                ...
        personnel-level-format/         ← persistent IDs across re-entries
        utilities.py

Usage:
    # default: evaluate on test split
    python run_trackers.py --data_root ../VIP-HTD

    # try a single sequence first to verify pipeline works
    python run_trackers.py --data_root ../VIP-HTD --max_seqs 1 --verbose
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_gt(gt_path: Path) -> dict:
    """Load MOTChallenge gt.txt -> {frame_id: [[x,y,w,h,track_id], ...]}"""
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            cls = int(parts[7]) if len(parts) > 7 else 1
            if cls != 1:
                continue
            gt.setdefault(frame_id, []).append([x, y, w, h, track_id])
    return gt


def save_mot_results(results: list, out_path: Path):
    """Save tracker output in MOTChallenge format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for frame_id, track_id, x, y, w, h, conf in results:
            f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")


def find_sequences(split_dir: Path):
    """Return sequence dirs under a split that contain img1/ and gt/gt.txt."""
    return sorted([
        d for d in split_dir.iterdir()
        if d.is_dir() and (d / "img1").exists() and (d / "gt" / "gt.txt").exists()
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Tracker instantiation
# ──────────────────────────────────────────────────────────────────────────────

def make_tracker(name: str, args, device: torch.device):
    if name == "bytetrack":
        return ByteTrack(
            min_conf=args.conf,
            track_thresh=0.45,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30,
        )
    if name == "strongsort":
        return StrongSort(
            reid_weights=Path(args.reid_weights),  # auto-downloaded by boxmot if missing
            device=device,
            half=False,
            min_conf=args.conf,
            max_cos_dist=0.3,
            max_iou_dist=0.7,
            n_init=3,
            nn_budget=100,
        )
    raise ValueError(f"Unknown tracker: {name}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-sequence tracking loop
# ──────────────────────────────────────────────────────────────────────────────

def run_tracker_on_sequence(tracker, detector: YOLO, seq_dir: Path,
                            conf_threshold: float, verbose: bool = False) -> list:
    img_dir = seq_dir / "img1"
    frames = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not frames:
        print(f"  [!] no frames in {img_dir}")
        return []

    results = []
    for frame_idx, frame_path in enumerate(frames):
        frame_id = frame_idx + 1  # 1-indexed
        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            continue

        # YOLO detect
        yolo_out = detector(img, verbose=False, conf=conf_threshold)[0]
        boxes = yolo_out.boxes
        if boxes is None or len(boxes) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            mask = boxes.cls == 0  # COCO person class
            if mask.sum() == 0:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                xyxy = boxes.xyxy[mask].cpu().numpy()
                confs = boxes.conf[mask].cpu().numpy().reshape(-1, 1)
                cls = boxes.cls[mask].cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, confs, cls])

        # track
        tracks = tracker.update(dets, img)
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                track_id = int(t[4])
                conf = float(t[5]) if len(t) > 5 else 1.0
                results.append((frame_id, track_id, x1, y1, x2 - x1, y2 - y1, conf))

        if verbose and frame_id % 200 == 0:
            n_tracks = 0 if tracks is None else len(tracks)
            print(f"    frame {frame_id}/{len(frames)}, tracks={n_tracks}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight metrics
#
# IDSW: count of times a GT identity gets matched to a different predicted ID
#       than it was previously matched to.
# MOTA: 1 - (FN + FP + IDSW) / GT
# IDF1: requires bipartite global matching; computed via TrackEval below
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics_simple(gt_all: dict, pred_all: dict, iou_threshold: float = 0.5) -> dict:
    total_idsw, total_gt, total_fp, total_tp = 0, 0, 0, 0

    for seq, gt_seq in gt_all.items():
        pred_seq = pred_all.get(seq, {})
        gt_to_pred = {}  # last-known GT_id -> pred_id within this sequence

        for frame_id in sorted(gt_seq.keys()):
            gt_dets = gt_seq[frame_id]
            pred_dets = pred_seq.get(frame_id, [])
            total_gt += len(gt_dets)

            matched_pred = set()
            for gt_box in gt_dets:
                gx, gy, gw, gh, gt_id = gt_box[:5]
                gx2, gy2 = gx + gw, gy + gh

                best_iou, best_idx, best_pred_id = 0, None, None
                for idx, pred_box in enumerate(pred_dets):
                    if idx in matched_pred:
                        continue
                    px, py, pw, ph, pred_id = pred_box[:5]
                    px2, py2 = px + pw, py + ph
                    inter = max(0, min(gx2, px2) - max(gx, px)) * max(0, min(gy2, py2) - max(gy, py))
                    union = gw * gh + pw * ph - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_idx, best_pred_id = iou, idx, pred_id

                if best_iou >= iou_threshold and best_pred_id is not None:
                    total_tp += 1
                    matched_pred.add(best_idx)
                    prev = gt_to_pred.get(int(gt_id))
                    if prev is not None and prev != best_pred_id:
                        total_idsw += 1
                    gt_to_pred[int(gt_id)] = best_pred_id

            total_fp += len(pred_dets) - len(matched_pred)

    fn = total_gt - total_tp
    mota = 1 - (fn + total_fp + total_idsw) / total_gt if total_gt > 0 else 0
    return {
        "IDSW": total_idsw,
        "MOTA": round(mota * 100, 2),
        "Precision": round(100 * total_tp / max(total_tp + total_fp, 1), 2),
        "Recall": round(100 * total_tp / max(total_gt, 1), 2),
        "TP": total_tp,
        "FP": total_fp,
        "FN": fn,
        "GT": total_gt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="../VIP-HTD",
                   help="Path to VIP-HTD root (the folder containing mot-challenge-format/)")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "test", "validation"],
                   help="Which VIP-HTD split to evaluate on")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--detector", type=str, default="yolov8m.pt",
                   help="YOLO weights (use a fine-tuned hockey checkpoint later)")
    p.add_argument("--reid_weights", type=str, default="osnet_x0_25_msmt17.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trackers", nargs="+", default=["bytetrack", "strongsort"],
                   choices=["bytetrack", "strongsort"])
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    split_dir = data_root / "mot-challenge-format" / args.split
    output_dir = Path(args.output_dir) / args.split
    device = torch.device(args.device)

    if not split_dir.exists():
        print(f"[ERROR] split dir not found: {split_dir}")
        print(f"        expected VIP-HTD/mot-challenge-format/{args.split}/")
        sys.exit(1)

    sequences = find_sequences(split_dir)
    if args.max_seqs:
        sequences = sequences[:args.max_seqs]

    if not sequences:
        print(f"[ERROR] no sequences found under {split_dir}")
        sys.exit(1)

    print(f"Device      : {device}")
    print(f"Split       : {args.split}")
    print(f"Sequences   : {len(sequences)}")
    print(f"Detector    : {args.detector}")
    for s in sequences:
        print(f"  - {s.name}")

    # detector
    print(f"\nLoading detector {args.detector}...")
    detector = YOLO(args.detector)

    # ground truth
    print("Loading ground truth...")
    gt_all = {}
    for seq in sequences:
        gt = load_gt(seq / "gt" / "gt.txt")
        gt_all[seq.name] = gt
        n_ids = len({int(b[4]) for f in gt.values() for b in f})
        print(f"  {seq.name}: {len(gt)} frames, {n_ids} unique IDs")

    # run trackers
    summary = {}
    for tracker_name in args.trackers:
        print(f"\n{'='*60}\nTracker: {tracker_name.upper()}\n{'='*60}")
        pred_all = {}
        t0 = time.time()

        for seq in sequences:
            print(f"\n  {seq.name}")
            tracker = make_tracker(tracker_name, args, device)
            seq_results = run_tracker_on_sequence(
                tracker=tracker, detector=detector, seq_dir=seq,
                conf_threshold=args.conf, verbose=args.verbose,
            )
            out_path = output_dir / tracker_name / f"{seq.name}.txt"
            save_mot_results(seq_results, out_path)
            print(f"    -> {len(seq_results)} dets, saved to {out_path}")

            pred_frames = {}
            for fid, tid, x, y, w, h, c in seq_results:
                pred_frames.setdefault(fid, []).append([x, y, w, h, tid, c])
            pred_all[seq.name] = pred_frames

        elapsed = time.time() - t0
        metrics = compute_metrics_simple(gt_all, pred_all)
        metrics["Time(s)"] = round(elapsed, 1)
        summary[tracker_name] = metrics

        print(f"\n  {'─'*40}")
        for k, v in metrics.items():
            print(f"    {k:<12}: {v}")

    # summary table
    print(f"\n{'='*60}\nSUMMARY ({args.split} split)\n{'='*60}")
    print(f"{'Tracker':<12} {'MOTA':>8} {'IDSW':>8} {'Prec':>8} {'Rec':>8} {'Time(s)':>10}")
    print("-" * 60)
    for name, m in summary.items():
        print(f"{name:<12} {m['MOTA']:>8} {m['IDSW']:>8} "
              f"{m['Precision']:>8} {m['Recall']:>8} {m['Time(s)']:>10}")

    print(f"\n[Note] IDF1 needs TrackEval's global bipartite matching.")
    print(f"       Tracker outputs are saved in MOTChallenge format under {output_dir}")
    print(f"       Run TrackEval separately for precise IDF1.")


if __name__ == "__main__":
    main()