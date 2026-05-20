from ultralytics import YOLO
from utils.eval_utils import find_sequences
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.reid import ReID
from DinoReID import DinoReID
from utils.eval_utils import load_gt, compute_metrics, crop_and_letterbox_square
import torch

def run_tracker(frame_paths, tracker, detector, custom_emb_model=None):
    # simulate getting one frame at a time
    tracks = None
    results = [] # list of tuples -> (frame_id, track_id, x1, y1, w, h, conf)
    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx % 100 == 0:
            print(f"on frame {frame_idx}")
        frame_id = frame_idx + 1
        frame = cv2.imread(str(frame_path)) # bgr
        # get bboxes from yolo
        dets = detector(frame, verbose=False)[0]
        dets_data = dets.boxes.data.cpu().numpy()
            
        if custom_emb_model is not None:
            bboxes = dets.boxes.xyxy.cpu().numpy()
            crops = [] # crops are bgr
            for x1, y1, x2, y2 in bboxes:
                crop = crop_and_letterbox_square(frame, [x1, y1, x2, y2])
                #crop = frame[int(y1):int(y2), int(x1):int(x2)]
                crops.append(crop)
            # run dino (or other embedding models) on crops to get embeddings
            embeddings = custom_emb_model(crops)
            if len(embeddings) != len(dets_data):
                print("embeddings and detections not same length!")

            # update tracker
            tracks = tracker.update(dets_data, frame, embs=embeddings)
        else:
            tracks = tracker.update(dets_data, frame)
        
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                track_id = int(t[4])
                conf = float(t[5]) if len(t) > 5 else 1.0
                results.append((frame_id, track_id, x1, y1, x2 - x1, y2 - y1, conf))
    return results


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    yolo_path = "yolo_trained/yolo11m_vip/weights/best.pt"
    seq_path = "VIP-HTD/mot-challenge-format/test"
    detector = YOLO(yolo_path)
    dino_reid = DinoReID(device=device)

    osnet_model = ReID(
        weights="osnet_x0_25_msmt17.pt",
        device=device,
        half=False
    )

    seqs = find_sequences(seq_path)
    seqs = [seqs[0]]

    total_gt = {}
    for seq in seqs:
        gt = load_gt(seq / "gt" / "gt.txt") # each gt is dict of lists of lists
        total_gt[seq.name] = gt

    total_preds = {} # dict of dicts of lists of lists
    # each sequence gets a dict, each frame is a key in a dict, each value is a list of track detection (bbox + track id)
    print("tracking...")
    for seq in seqs:
        print(f"on seq {seq.name}")
        img_dir = seq / "img1"
        frame_paths = sorted(img_dir.glob("*.jpg"))
        # fresh tracker per sequence bc strongsort holds internal state
        tracker = StrongSort(reid_model=osnet_model.model)
        tracker.update
        seq_results = run_tracker(frame_paths, tracker, detector, custom_emb_model=dino_reid)

        curr_preds = {}
        for fid, tid, x, y, w, h, c in seq_results:
            curr_preds.setdefault(fid, []).append([tid, x, y, w, h])
        total_preds[seq.name] = curr_preds
    print("done tracking!")
    
    metrics = compute_metrics(total_gt, total_preds, iou_thresh=0.3) # tune iou thresh!
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # TODO: 1. re-run metrics with to calc idf1
    # 1.25 find good padding value - its at image net mean rn (but was black before - could try white too since same color as ice?)
    # 1.5 try diff dino sizes (vits, vitb, vitl, vitg) - maybe finetune vits or vitb, bc others too large
    # 2. tune iou_thresh (do a sweep?) - since gt bboxes are loose/not super accurate
    # 3. tune max_cos_dist for dino (apparently cosine distance distribution diff for diff embeddings?)
    # 3.5 try diff cropping techniques for dino? is current one with padding the best? (i think so tbh, or maybe pad with 255 since white is like ice)
    # 4. run metrics on entire test set (call test sequences)
    # 5. finetune dino with constrastive learning and see what results we can get (figure out which dino to finetune)
    # 6. can try finetuning just the projection head, or potentially also the last 1-2 transformer blocks - could do ablation!
    # when finetuning, do hard on negatives of players of same team 

    

if __name__ == "__main__":
    main()