from ultralytics import YOLO
from utils.eval_utils import find_sequences
import cv2


def main():
    yolo_path = "yolo_trained/yolo11m_vip/weights/best.pt"
    seq_path = "VIP-HTD/mot-challenge-format/test"
    detector = YOLO(yolo_path)
    tracker = PlayerTracker()
    seqs = find_sequences(seq_path)
    test_seq = seqs[0]
    img_dir = test_seq / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))

    # simulate getting one frame at a time
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        # get bboxes from yolo
        # run dino on crops to get embeddings
        # call tracker.update(detections, frame, embeddings)
    

if __name__ == "__main__":
    main()