from pathlib import Path
import configparser

# gt.txt -> {frame_id: [[,track_id,x,y,w,h], ...]}
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
    