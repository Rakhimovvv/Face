"""
O'quv ma'lumotlari (masalan, UTKFace) bo'yicha yosh regressiyasini o'qitish va
`checkpoints/age_predictor.onnx` faylini yaratish.

Bu skript diplomning «model qurish / o'qitish» qismi.
Ishga tushirish uchun odatda alohida muhit kerak (Python 3.10–3.11 + PyTorch),
chunki ba'zi tizimlarda Python 3.14 uchun torch barqaror bo'lmasligi mumkin.

O'rnatish (o'qitish muhiti):
    pip install torch torchvision

UTKFace: rasmlar bitta papkada, fayl nomi [yosh]_[jins]_... bilan boshlanadi.

Misol:
    python train_age.py --data_dir "D:\\data\\UTKFace\\train" --epochs 20 --batch_size 32

Natija: checkpoints/best.pth va checkpoints/age_predictor.onnx
Keyin: python app.py  (onnxruntime bilan faqat .onnx ishlatiladi)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


# --- O'zingizning arxitekturangiz (tayyor Caffe vazn emas) ---
class FaceAgeNet(nn.Module):
    """Yuz qirqindisi uchun engil CNN + regressiya boshligi."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)


def _parse_age_utk(path: Path) -> int | None:
    """UTKFace: birinchi token — yosh (butun son)."""
    part = path.name.split("_")[0]
    try:
        a = int(part)
    except ValueError:
        return None
    if a < 0 or a > 120:
        return None
    return a


class UTKFaceFolder(Dataset):
    def __init__(
        self,
        root: Path,
        size: int = 224,
        max_samples: int = 0,
        seed: int = 42,
    ) -> None:
        self.root = root = Path(root).expanduser().resolve()
        self.size = size

        if not root.is_dir():
            raise FileNotFoundError(
                f"Papka topilmadi yoki katalog emas: {root}\n\n"
                "docstringdagi «SIZNING_UTKFACE_PAPKANGIZ» — namuna, uni yozmaydi.\n"
                "UTKFace ni yuklab oling va haqiqiy yo'lni bering, masalan:\n"
                r'  python train_age.py --data_dir "D:\data\UTKFace_aligned_cropped\UTKFace"'
            )

        paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        samples: list[tuple[Path, int]] = []
        for p in paths:
            age = _parse_age_utk(p)
            if age is not None:
                samples.append((p, age))
        if not samples:
            n_img = len(paths)
            ex = f"  Masalan to'g'ri nom: 25_1_0_20160116174525125.jpg\n" if n_img == 0 else (
                f"  Topilgan birinchi fayl: {paths[0].name}\n"
                "  UTKFace: nom `YOSH_butun_son_jins_...` bilan boshlanadi (birinchi '_' oldigacha yosh).\n"
            )
            raise FileNotFoundError(
                f"{root} ichida UTKFace qoidasiga mos rasm yo'q "
                f"(jpg/jpeg/png jami: {n_img}).\n\n"
                f"{ex}"
                "Yanada ko'p ma'lumot: `utkface` dataset — `aligned and cropped` variantdan foydalaning."
            )

        rng = random.Random(seed)
        rng.shuffle(samples)
        if max_samples and max_samples < len(samples):
            samples = samples[:max_samples]
        self.samples = samples

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, age = self.samples[i]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        t = t.permute(2, 0, 1)
        t = (t - self._mean) / self._std
        return t, torch.tensor(float(age), dtype=torch.float32)


def export_face_age_net_onnx(model_cpu: FaceAgeNet, onnx_path: Path) -> None:
    """TorchScript asosidagi eksport; dynamo=False — onnxscript importidan qochish."""
    model_cpu.eval()
    dummy = torch.randn(1, 3, 224, 224)
    path_str = str(onnx_path.resolve())
    daxes = {"input": {0: "batch"}, "age": {0: "batch"}}
    try:
        torch.onnx.export(
            model_cpu,
            dummy,
            path_str,
            input_names=["input"],
            output_names=["age"],
            dynamic_axes=daxes,
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model_cpu,
            dummy,
            path_str,
            input_names=["input"],
            output_names=["age"],
            dynamic_axes=daxes,
            opset_version=17,
        )


def export_only_from_checkpoint(out_dir: Path) -> None:
    """O'qitishni qayta boshlamasdan faqat age_predictor.onnx yaratish."""
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pth"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"{best_path} topilmadi. Avval o'qitishni ishga tushiring yoki --out_dir ni tekshiring."
        )
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location="cpu")
    model_cpu = FaceAgeNet()
    model_cpu.load_state_dict(ckpt["model"])
    onnx_path = out_dir / "age_predictor.onnx"
    export_face_age_net_onnx(model_cpu, onnx_path)
    mae = ckpt.get("mae", "?")
    print(f"ONNX eksport: {onnx_path.resolve()}")
    print(f"Checkpoint MAE (yil): {mae}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="UTKFace rasmlari papkasi (--export_only bo'lmasa majburiy)",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--max_samples", type=int, default=0, help="0 = barchasi; tez sinov uchun cheklang")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--export_only",
        action="store_true",
        help="Faqat best.pth dan age_predictor.onnx (qayta o'qitmaydi)",
    )
    args = ap.parse_args()

    if args.export_only:
        export_only_from_checkpoint(args.out_dir)
        return

    if args.data_dir is None:
        raise SystemExit("--data_dir majburiy (--export_only bo'lmasa).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = UTKFaceFolder(
        args.data_dir,
        max_samples=args.max_samples or 0,
        seed=args.seed,
    )
    n_val = max(1, int(0.2 * len(full)))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FaceAgeNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    best_mae = float("inf")
    best_path = args.out_dir / "best.pth"

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        abs_sum = 0.0
        n_v = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                abs_sum += (pred - y).abs().sum().item()
                n_v += x.size(0)
        mae = abs_sum / max(1, n_v)

        print(f"epoch {ep}/{args.epochs}  train_L1={tr_loss:.3f}  val_MAE_yil={mae:.3f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({"model": model.state_dict(), "mae": mae}, best_path)
            print(f"  -> saqladik {best_path}")

    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:  # eski PyTorch
        ckpt = torch.load(best_path, map_location="cpu")
    model_cpu = FaceAgeNet()
    model_cpu.load_state_dict(ckpt["model"])
    model_cpu.eval()

    onnx_path = args.out_dir / "age_predictor.onnx"
    export_face_age_net_onnx(model_cpu, onnx_path)
    print(f"ONNX eksport: {onnx_path.resolve()}")
    print(f"Eng yaxshi val_MAE (yil): {best_mae:.3f} — buni diplomda jadval sifatida keltiring.")


if __name__ == "__main__":
    _main()
