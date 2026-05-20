from pathlib import Path
import configparser
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

# gt.txt -> {frame_id: [[track_id,x,y,w,h], ...]}
def load_gt(gt_path):
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(int, parts[2:6])
            gt.setdefault(frame_id, []).append([track_id, x, y, w, h])
    return gt

def find_sequences(split_dir):
    # Return sequence dirs under a split that contain img1/ and gt/gt.txt.
    split_dir = Path(split_dir)
    return sorted([
        d for d in split_dir.iterdir()
        if d.is_dir() and (d / "img1").exists() and (d / "gt" / "gt.txt").exists()
    ])

def load_ini(ini_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    h = config["Sequence"].getint("imHeight")
    w = config["Sequence"].getint("imWidth")
    return w, h

# returns path to .txt file, listing abs path of all images and labels
# also symlinks img1 to images for yolo
def prepare_split_paths(seq_path):
    # symlink images, write a .txt list of image paths for this split
    sequences = find_sequences(seq_path)
    # ultralytics finds labels by swapping /images/ -> /labels/ in the
    # image path, so expose img1 as `images` per sequence.
    for seq in sequences:
        link = seq / "images"
        if not link.is_symlink() and not link.exists():
            link.symlink_to("img1")
    # absolute paths via the `images` symlink so the /images/ -> /labels/
    # swap fires. .absolute() does not resolve symlinks; .resolve() would.
    list_path = Path(f"/tmp/vip_{seq_path.name}.txt")
    with open(list_path, "w") as f:
        for seq in sequences:
            for img in sorted((seq / "images").glob("*.jpg")):
                f.write(f"{img.absolute()}\n")
    return list_path

def _iou(g, p):
    _, gx, gy, gw, gh = g[:5]
    _, px, py, pw, ph = p[:5]
    gx2, gy2, px2, py2 = gx + gw, gy + gh, px + pw, py + ph
    inter = max(0, min(gx2, px2) - max(gx, px)) * max(0, min(gy2, py2) - max(gy, py))
    union = gw * gh + pw * ph - inter
    return inter / union if union > 0 else 0

def compute_metrics(gt, pred, iou_thresh=0.5):
    total_idsw, total_gt, total_pred, total_fp, total_tp = 0, 0, 0, 0, 0

    # For IDF1
    # seq -> {(gt_id, pred_id): number of frame-level matches}
    id_match_counts = defaultdict(lambda: defaultdict(int))
    gt_ids_by_seq = defaultdict(set)
    pred_ids_by_seq = defaultdict(set)

    # Include predicted-only sequences too, so extra predictions count as FP/IDFP
    all_seqs = sorted(set(gt.keys()) | set(pred.keys()))

    for seq in all_seqs:
        gt_seq = gt.get(seq, {})
        pred_seq = pred.get(seq, {})
        gt_to_pred = {}  # last-known GT_id -> pred_id within this sequence

        all_frames = sorted(set(gt_seq.keys()) | set(pred_seq.keys()))

        for frame_id in all_frames:
            gt_dets = gt_seq.get(frame_id, [])
            pred_dets = pred_seq.get(frame_id, [])

            total_gt += len(gt_dets)
            total_pred += len(pred_dets)

            for g in gt_dets:
                gt_ids_by_seq[seq].add(int(g[0]))

            for p in pred_dets:
                pred_ids_by_seq[seq].add(int(p[0]))

            if len(gt_dets) == 0:
                total_fp += len(pred_dets)
                continue

            if len(pred_dets) == 0:
                continue

            # Build IoU matrix and solve optimal frame-level assignment
            iou_mat = np.array([[_iou(g, p) for p in pred_dets] for g in gt_dets])
            gt_inds, pred_inds = linear_sum_assignment(-iou_mat)

            matched_pred = set()

            for gi, pi in zip(gt_inds, pred_inds):
                if iou_mat[gi, pi] < iou_thresh:
                    continue

                total_tp += 1
                matched_pred.add(pi)

                gt_id = int(gt_dets[gi][0])
                pred_id = int(pred_dets[pi][0])

                # IDSW
                prev = gt_to_pred.get(gt_id)
                if prev is not None and prev != pred_id:
                    total_idsw += 1
                gt_to_pred[gt_id] = pred_id

                # IDF1 pairwise identity overlap count
                id_match_counts[seq][(gt_id, pred_id)] += 1

            total_fp += len(pred_dets) - len(matched_pred)

    # -------------------------
    # IDF1 global ID assignment
    # -------------------------
    idtp = 0

    for seq in all_seqs:
        gt_ids = sorted(gt_ids_by_seq[seq])
        pred_ids = sorted(pred_ids_by_seq[seq])

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            continue

        gt_id_to_row = {gid: i for i, gid in enumerate(gt_ids)}
        pred_id_to_col = {pid: j for j, pid in enumerate(pred_ids)}

        score_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

        for (gt_id, pred_id), count in id_match_counts[seq].items():
            r = gt_id_to_row[gt_id]
            c = pred_id_to_col[pred_id]
            score_mat[r, c] = count

        rows, cols = linear_sum_assignment(-score_mat)

        for r, c in zip(rows, cols):
            idtp += int(score_mat[r, c])

    idfp = total_pred - idtp
    idfn = total_gt - idtp

    id_precision = idtp / max(idtp + idfp, 1)
    id_recall = idtp / max(idtp + idfn, 1)
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    fn = total_gt - total_tp
    mota = 1 - (fn + total_fp + total_idsw) / total_gt if total_gt > 0 else 0

    return {
        "IDSW": total_idsw,
        "MOTA": round(mota * 100, 2),
        "IDF1": round(idf1 * 100, 2),
        "IDP": round(id_precision * 100, 2),
        "IDR": round(id_recall * 100, 2),
        "Precision": round(100 * total_tp / max(total_tp + total_fp, 1), 2),
        "Recall": round(100 * total_tp / max(total_gt, 1), 2),
        "TP": total_tp,
        "FP": total_fp,
        "FN": fn,
        "GT": total_gt,
        "Pred": total_pred,
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
    }

def crop_and_letterbox_square(img, box, out_size=224, pad_value=(104, 116, 124)):
    """
    img: BGR image, H x W x 3
    box: [x1, y1, x2, y2]
    returns: square BGR crop, out_size x out_size
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, np.round(box))

    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return np.full((out_size, out_size, 3), pad_value, dtype=np.uint8)

    crop = img[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    side = max(h, w)

    square = np.full((side, side, 3), pad_value, dtype=np.uint8)

    top = (side - h) // 2
    left = (side - w) // 2

    square[top:top+h, left:left+w] = crop

    square = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return square
    