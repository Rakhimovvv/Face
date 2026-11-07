"""
O'quv ma'lumotlari (masalan, UTKFace) bo'yicha yosh bashorati modelini o'qitish va
`checkpoints/age_predictor.onnx` faylini yaratish.

Bu skript diplomning «model qurish / o'qitish» qismi.

Model: ImageNet'da oldindan o'qitilgan ResNet18 (transfer learning) + yosh boshligi.
Boshliq 0..100 yosh uchun ehtimollik taqsimotini chiqaradi (DLDL: Gauss yorliq
taqsimoti + kutilgan yosh bo'yicha L1). ONNX chiqishi — bitta son: kutilgan yosh.

O'rnatish (o'qitish muhiti):
    pip install -r requirements-train.txt

UTKFace: rasmlar bitta papkada, fayl nomi [yosh]_[jins]_... bilan boshlanadi.

Misol:
    python train_age.py --data_dir "C:\\Users\\...\\UTKFace" --epochs 8

Uzilib qolsa (CPU da bir epoxa ~20-30 daqiqa), xuddi shu argumentlar + --resume:
    python train_age.py --data_dir "..." --epochs 8 --resume

Natija: checkpoints/best.pth, last.pth, train_log.txt, history.csv va age_predictor.onnx
Keyin: python app.py  (onnxruntime bilan faqat .onnx ishlatiladi)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError as e:
    raise SystemExit(
        "PyTorch (torch, torchvision) topilmadi — train_age.py uchun zarur.\n\n"
        "Loyiha papkasida:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements-train.txt\n"
        "  .\\.venv\\Scripts\\python.exe train_age.py --export_only\n"
    ) from e

from PIL import Image

MAX_AGE = 100
ARCH = "resnet18_dldl"
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
# Natijalarni yosh guruhlari bo'yicha ko'rsatish uchun (pastki chegara, yuqori chegara)
AGE_GROUPS = [(0, 12), (13, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, MAX_AGE)]


class AgeNet(nn.Module):
    """ResNet18 asos + 0..MAX_AGE yosh sinflari; forward kutilgan yoshni ham beradi."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        n_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_feat, MAX_AGE + 1))
        self.register_buffer("ages", torch.arange(MAX_AGE + 1, dtype=torch.float32))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def expected_age(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=1) * self.ages).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expected_age(self.logits(x))


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


def _list_samples(root: Path) -> list[tuple[Path, int]]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Papka topilmadi yoki katalog emas: {root}\n"
            "UTKFace ni yuklab oling va haqiqiy yo'lni bering, masalan:\n"
            r'  python train_age.py --data_dir "D:\data\UTKFace"'
        )
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    samples = [(p, a) for p in paths if (a := _parse_age_utk(p)) is not None]
    if not samples:
        raise FileNotFoundError(
            f"{root} ichida UTKFace qoidasiga mos rasm yo'q (jpg/jpeg/png jami: {len(paths)}).\n"
            "  Masalan to'g'ri nom: 25_1_0_20160116174525125.jpg"
        )
    return samples


class AgeDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], tfm: transforms.Compose) -> None:
        self.samples = samples
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, age = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), torch.tensor(float(min(age, MAX_AGE)), dtype=torch.float32)


def _build_transforms(size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # Augmentatsiya: ilovadagi Haar qutisi o'lchami/joyi biroz tebranadi — masshtab va
    # siljish shunga chidamli qiladi; rang/yorug'lik — kamera sharoitlari uchun.
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(12)], p=0.5),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )
    val = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )
    return train, val


def _soft_labels(age: torch.Tensor, sigma: float) -> torch.Tensor:
    """Har bir yosh uchun Gauss yorliq taqsimoti (qo'shni yoshlar ham qisman to'g'ri)."""
    ages = torch.arange(MAX_AGE + 1, dtype=torch.float32, device=age.device)
    d = torch.exp(-((ages.unsqueeze(0) - age.unsqueeze(1)) ** 2) / (2 * sigma**2))
    return d / d.sum(dim=1, keepdim=True)


def _predict(model: AgeNet, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Val bashoratlari, gorizontal flip TTA bilan — ilovadagi bashorat bilan bir xil."""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, memory_format=torch.channels_last)
            preds.append((0.5 * (model(x) + model(torch.flip(x, dims=[3])))).cpu())
            ys.append(y)
    return torch.cat(preds), torch.cat(ys)


def _evaluate(model: AgeNet, loader: DataLoader, device: torch.device) -> float:
    """Val MAE (yil)."""
    pred, y = _predict(model, loader, device)
    return (pred - y).abs().mean().item()


def _group_report(model: AgeNet, loader: DataLoader, device: torch.device) -> list[str]:
    """Yosh guruhlari bo'yicha MAE — model qaysi yoshlarda ko'proq adashishini ko'rsatadi."""
    pred, y = _predict(model, loader, device)
    err = (pred - y).abs()
    lines = [f"  {'guruh':>8}  {'soni':>5}  {'MAE':>6}"]
    for lo, hi in AGE_GROUPS:
        m = (y >= lo) & (y <= hi)
        if m.any():
            lines.append(f"  {f'{lo}-{hi}':>8}  {int(m.sum()):>5}  {err[m].mean().item():>6.2f}")
    lines.append(f"  {'jami':>8}  {len(y):>5}  {err.mean().item():>6.2f}")
    return lines


def export_onnx(model_cpu: AgeNet, onnx_path: Path, img_size: int, meta: dict | None = None) -> None:
    """TorchScript asosidagi eksport; dynamo=False — onnxscript importidan qochish.

    meta — ONNX metadata_props ga yoziladi (ilova model nomi va MAE ni ko'rsatadi).
    """
    model_cpu.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs = dict(
        input_names=["input"],
        output_names=["age"],
        dynamic_axes={"input": {0: "batch"}, "age": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model_cpu, dummy, str(onnx_path.resolve()), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model_cpu, dummy, str(onnx_path.resolve()), **kwargs)
    if meta:
        import onnx

        m = onnx.load(str(onnx_path))
        for k, v in meta.items():
            p = m.metadata_props.add()
            p.key, p.value = k, str(v)
        onnx.save(m, str(onnx_path))


def _onnx_meta(ckpt: dict) -> dict:
    mae = ckpt.get("mae")
    return {
        "arch": ckpt.get("arch", ARCH),
        "img_size": ckpt.get("img_size", 160),
        "val_mae": f"{mae:.2f}" if isinstance(mae, (int, float)) else "?",
        "dataset": "UTKFace",
    }


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # eski PyTorch
        return torch.load(path, map_location="cpu")


def _save_atomic(obj: dict, path: Path) -> None:
    """Saqlash paytida uzilsa ham eski fayl buzilmaydi."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _load_checkpoint(best_path: Path) -> tuple[AgeNet, dict]:
    ckpt = _torch_load(best_path)
    if ckpt.get("arch") != ARCH:
        raise SystemExit(
            f"{best_path} eski arxitektura (FaceAgeNet) checkpointi. "
            "Yangi model uchun qayta o'qiting: python train_age.py --data_dir ..."
        )
    model = AgeNet(pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def export_only_from_checkpoint(out_dir: Path) -> None:
    """O'qitishni qayta boshlamasdan faqat age_predictor.onnx yaratish."""
    best_path = out_dir / "best.pth"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"{best_path} topilmadi. Avval o'qitishni ishga tushiring yoki --out_dir ni tekshiring."
        )
    model, ckpt = _load_checkpoint(best_path)
    onnx_path = out_dir / "age_predictor.onnx"
    export_onnx(model, onnx_path, int(ckpt.get("img_size", 160)), _onnx_meta(ckpt))
    print(f"ONNX eksport: {onnx_path.resolve()}")
    print(f"Checkpoint MAE (yil): {ckpt.get('mae', '?')}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=None, help="UTKFace rasmlari papkasi")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--sigma", type=float, default=2.0, help="Gauss yorliq kengligi (yil)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader jarayonlari (kam RAM bo'lsa 0-2)")
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--max_samples", type=int, default=0, help="0 = barchasi; tez sinov uchun cheklang")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--export_only",
        action="store_true",
        help="Faqat best.pth dan age_predictor.onnx (qayta o'qitmaydi)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="last.pth dan davom ettirish (argumentlar avvalgi ishga tushirish bilan bir xil bo'lsin)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        export_only_from_checkpoint(args.out_dir)
        return
    if args.data_dir is None:
        raise SystemExit("--data_dir majburiy (--export_only bo'lmasa).")

    log_path = args.out_dir / "train_log.txt"
    hist_path = args.out_dir / "history.csv"
    if not args.resume:
        log_path.unlink(missing_ok=True)
        hist_path.unlink(missing_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = _list_samples(args.data_dir)
    random.Random(args.seed).shuffle(samples)
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
    n_val = max(1, int(args.val_frac * len(samples)))
    val_s, train_s = samples[:n_val], samples[n_val:]
    log(f"Train: {len(train_s)}  Val: {len(val_s)}  device: {device}")

    tf_train, tf_val = _build_transforms(args.img_size)
    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        AgeDataset(train_s, tf_train), batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    # Val uchun alohida worker jarayonlari ochilmaydi: Windows'da har biri torch'ni qayta yuklab
    # xotirani band qiladi (8 GB RAM da o'qitish shu sababli yiqilgan), tezlikka esa deyarli ta'sir qilmaydi.
    val_loader = DataLoader(AgeDataset(val_s, tf_val), batch_size=128, shuffle=False, num_workers=0)

    # channels_last — CPU (oneDNN) da konvolyutsiyalar ~10% tezroq
    model = AgeNet(pretrained=True).to(device, memory_format=torch.channels_last)
    # Asos (backbone) uchun kichikroq lr — ImageNet bilimlarini buzmaslik uchun
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * 0.3},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * 0.3, args.lr], total_steps=steps, pct_start=0.15
    )

    best_mae = math.inf
    start_ep = 1
    best_path = args.out_dir / "best.pth"
    last_path = args.out_dir / "last.pth"
    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"{last_path} topilmadi — --resume siz boshlang.")
        st = _torch_load(last_path)
        if st.get("total_steps") != steps:
            raise SystemExit(
                "last.pth boshqa sozlamalar bilan yaratilgan: --epochs, --batch_size, --max_samples, "
                "--val_frac avvalgidek bo'lishi kerak."
            )
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"])
        start_ep, best_mae = st["epoch"] + 1, st["best_mae"]
        log(f"Davom ettirilmoqda: epoch {start_ep}/{args.epochs}, eng yaxshi val MAE {best_mae:.3f}")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_mae, n_seen = 0.0, 0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, memory_format=torch.channels_last), y.to(device)
            logits = model.logits(x)
            pred = model.expected_age(logits)
            kl = F.kl_div(F.log_softmax(logits, dim=1), _soft_labels(y, args.sigma), reduction="batchmean")
            loss = kl + 0.1 * F.l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tr_mae += (pred.detach() - y).abs().sum().item()
            n_seen += x.size(0)
            if i % 50 == 0:
                log(f"  ep{ep} {i}/{len(train_loader)}  train_MAE={tr_mae / n_seen:.2f}")

        mae = _evaluate(model, val_loader, device)
        tr = tr_mae / max(1, n_seen)
        secs = time.time() - t0
        log(f"epoch {ep}/{args.epochs}  train_MAE={tr:.3f}  val_MAE_yil={mae:.3f}  ({secs:.0f}s)")
        new_file = not hist_path.is_file()
        with hist_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["epoch", "train_mae", "val_mae", "seconds"])
            w.writerow([ep, f"{tr:.3f}", f"{mae:.3f}", f"{secs:.0f}"])
        if mae < best_mae:
            best_mae = mae
            _save_atomic(
                {"model": model.state_dict(), "mae": mae, "arch": ARCH, "img_size": args.img_size},
                best_path,
            )
            log(f"  -> saqladik {best_path}")
        _save_atomic(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "rng": torch.get_rng_state(),
                "epoch": ep,
                "best_mae": best_mae,
                "total_steps": steps,
            },
            last_path,
        )

    model_cpu, ckpt = _load_checkpoint(best_path)
    log("Eng yaxshi model, yosh guruhlari bo'yicha val MAE (yil):")
    for line in _group_report(model_cpu.to(device), val_loader, device):
        log(line)
    onnx_path = args.out_dir / "age_predictor.onnx"
    export_onnx(model_cpu.cpu(), onnx_path, args.img_size, _onnx_meta(ckpt))
    log(f"ONNX eksport: {onnx_path.resolve()}  (eng yaxshi val MAE: {best_mae:.2f} yil)")


if __name__ == "__main__":
    _main()
