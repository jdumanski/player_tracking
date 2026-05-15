import numpy as np
import csv
import os
from utils.eval_utils import load_gt, find_sequences, load_ini

# given path to sequences, for each sequence to converts gt.txt labels to labels/<frame_name>.txt for each frame for yolo
seq_path = "VIP-HTD/mot-challenge-format/validation"
seqs = find_sequences(seq_path)

for seq in seqs:
    gt_path = seq / "gt/gt.txt"
    gt = load_gt(gt_path)
    labels_path = seq / "labels"
    os.makedirs(labels_path, exist_ok=True)
    w, h = load_ini(seq / "seqinfo.ini")
    for frame_id, dets in gt.items():
        frame_name = str(frame_id).zfill(6) + ".txt"
        out_txt = labels_path / frame_name
        with open(out_txt, "w") as f:
            for det in dets:
                _, x, y, bw, bh = det # _ is track_id here
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                # first is 0 because person class is 0 here
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    