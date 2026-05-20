from pathlib import Path

import modal

app = modal.App("vip-htd-finetune")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "numpy")
    .add_local_python_source("utils")
)

vol = modal.Volume.from_name("vip-htd")

SPLITS = ["train", "validation", "test"]


def _do_train(data_root: Path, runs_root: Path, *,
              epochs: int, imgsz: int, batch: int, fraction: float | None,
              on_epoch_end=None):
    from ultralytics import YOLO
    from utils.eval_utils import prepare_split_paths

    paths = {s: prepare_split_paths(data_root / s) for s in SPLITS}
    yaml_path = Path("/tmp/vip_htd.yaml")
    yaml_path.write_text(
        f"train: {paths['train']}\n"
        f"val:   {paths['validation']}\n"
        f"test:  {paths['test']}\n"
        f"names:\n  0: player\n"
    )

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=10,
        project=str(runs_root),
        name="yolo11m_vip",
        workers=16,
        cache="ram",
    )
    if fraction is not None:
        train_kwargs["fraction"] = fraction

    model = YOLO("yolo11m.pt")
    if on_epoch_end is not None:
        model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.train(**train_kwargs)
    model.val(data=str(yaml_path), split="test", batch=32, workers=16)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    timeout=60 * 60 * 8,
)
def train_remote(epochs: int = 50, imgsz: int = 1280,
                 batch: int = 24, fraction: float | None = None):
    def commit_epoch(trainer):
        vol.commit()  # persist best.pt/last.pt after each fit epoch

    _do_train(
        data_root=Path("/vol/VIP-HTD/mot-challenge-format"),
        runs_root=Path("/vol/runs/finetune"),
        epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction,
        on_epoch_end=commit_epoch,
    )
    vol.commit()


def train_local(epochs: int = 1, imgsz: int = 640,
                batch: int = 4, fraction: float | None = 0.005):
    _do_train(
        data_root=Path("VIP-HTD/mot-challenge-format"),
        runs_root=Path("runs/finetune"),
        epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction,
    )


@app.local_entrypoint()
def main():
    train_remote.spawn()


if __name__ == "__main__":
    train_local()
