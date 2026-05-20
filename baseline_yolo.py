import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.eval_utils import find_sequences, load_gt, prepare_split_paths

# mot gt is frame_id, track_id, x, y, w, h, confidence, class, visibility
def viz_results(seq_path, detector, num_frames, out_folder, stride=1):
    yolo_person_cls_id = 0
    sequences = find_sequences(seq_path)
    results_root = Path(out_folder)
    GT_COLOR = (0, 255, 0)    # green
    PRED_COLOR = (0, 0, 255)  # red
    
    for i, seq in enumerate(sequences, start=1):
        img_dir = seq / "img1"
        gt = load_gt(seq / "gt/gt.txt")
        frames = sorted(img_dir.glob("*.jpg"))
        out_dir = results_root / seq.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"On sequence {i}: {seq.name}")
        curr_num_frames = min(num_frames*stride, len(frames))
        for frame_idx, frame_path in enumerate(frames[:curr_num_frames]):
            if frame_idx % stride != 0:
                continue
            img = cv2.imread(str(frame_path))

            # predictions (xyxy, person class only)
            dets = detector(img, verbose=False)[0]
            if len(dets.boxes) > 0:
                mask = dets.boxes.cls == yolo_person_cls_id
                coords = dets.boxes.xyxy[mask].cpu().numpy()
                confs = dets.boxes.conf[mask].cpu().numpy()
            else:
                coords = np.zeros((0, 4))
                confs = np.zeros(0)

            # MOT gt.txt is 1-indexed and image stems match (000001.jpg)
            mot_frame_id = int(frame_path.stem)
            frame_gt = gt.get(mot_frame_id, [])

            for _, x, y, w, h in frame_gt:
                cv2.rectangle(img, (int(x), int(y)),
                              (int(x + w), int(y + h)), GT_COLOR, 2)

            for (x1, y1, x2, y2), conf in zip(coords, confs):
                p1 = (int(x1), int(y1))
                cv2.rectangle(img, p1, (int(x2), int(y2)), PRED_COLOR, 2)
                cv2.putText(img, f"{conf:.2f}", (p1[0], max(p1[1] - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1, cv2.LINE_AA)

            cv2.putText(img, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GT_COLOR, 2)
            cv2.putText(img, "Pred", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PRED_COLOR, 2)

            out_path = out_dir / frame_path.name
            cv2.imwrite(str(out_path), img)
            #print(f"  saved {out_path}")

# run one model.val() over all sequences (micro-averaged metrics).
# one class true means model only have one class
def eval_detection_metrics(seq_path, model, one_class=False):
    seq_path = Path(seq_path)
    list_path = prepare_split_paths(seq_path) # path to .txt listing abs path of all images and labels

    yaml_path = Path("/tmp/vip.yaml")
    yaml_path.write_text(
        f"train: {list_path}\nval: {list_path}\nnames:\n  0: person\n"
    )
    if one_class:
        res = model.val(data=str(yaml_path), classes=[0], verbose=False)
    else:
        res = model.val(data=str(yaml_path), verbose=False)
    print(
        f"mAP50={res.box.map50:.4f} mAP50-95={res.box.map:.4f} "
        f"P={res.box.mp:.4f} R={res.box.mr:.4f}"
    )

def main():
    seq_path = "VIP-HTD/mot-challenge-format/test"
    ft_path = "./yolo_trained/yolo11m_vip/weights/best.pt"
    base_path = "./yolo11m.pt"
    model = YOLO(ft_path)
    viz_results(seq_path, model, 100, "detection_results/visualized/base", stride=100)
    #eval_detection_metrics(seq_path, model, one_class=True)
    
if __name__ == "__main__":
    main()