"""
Yuz tasviri bo'yicha yosh bashorati: real vaqt + rasm yuklash.

Tez lokal (standart): checkpoints/age_predictor.onnx — birinchi navbatda ishlatiladi.

Ixtiyoriy bulut: AGE_BACKEND=azure + AZURE_FACE_KEY / AZURE_FACE_ENDPOINT.

Yuz: ONNX rejimida OpenCV YuNet (yo'q bo'lsa Haar); Azure da — API ichki deteksiya.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "checkpoints" / "age_predictor.onnx"

_DEFAULT_IMG_SIZE = 224  # ONNX kirishida o'lcham yozilmagan bo'lsa
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Yuz detektori: YuNet (OpenCV DNN, qiyshiq/yorug'lik farqli yuzlarda ham ishonchli).
# Fayl yo'q bo'lsa — Haar kaskadiga qaytiladi.
YUNET_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SCORE = 0.6
YUNET_MAX_SIDE = 640  # katta rasmlarni deteksiya uchun kichraytirish (tezlik)

# Model UTKFace qirqimida o'qitilgan; detektor qutisini shunga moslash muhim.
# (kvadrat tomoni = scale * quti_kengligi, markaz pastga shift_y * quti_balandligi)
# Qiymatlar UTKFace rasmlarida har bir detektor uchun o'lchangan.
CROP_PARAMS = {"yunet": (1.19, 0.11), "haar": (1.21, 0.08)}

# Kichikroq = tezroq yangilanish (CPU yuklari ortishi mumkin)
STREAM_ANALYZE_INTERVAL_SEC = 0.3
# Kamerada natija sakramasligi uchun eksponensial silliqlash (0..1, katta = tezroq o'zgaradi)
STREAM_EMA_ALPHA = 0.35
# Yuz shuncha vaqt ko'rinmasa — silliqlash qaytadan boshlanadi (boshqa odam bo'lishi mumkin)
STREAM_RESET_AFTER_SEC = 1.5


def _new_stream_state() -> dict[str, Any]:
    """Kamera holati — har bir brauzer sessiyasi uchun alohida (gr.State)."""
    return {
        "last_analyze": 0.0,
        "last_face": 0.0,
        "age": None,
        "age_ema": None,
        "detail": "",
        "box": None,
    }


_face_cascade: cv2.CascadeClassifier | None = None
_yunet: Any = None
_yunet_failed = False
_ort_session: ort.InferenceSession | None = None
_ort_input_name: str | None = None
_img_size: int = _DEFAULT_IMG_SIZE


def _azure_configured() -> bool:
    key = os.environ.get("AZURE_FACE_KEY", "").strip()
    ep = os.environ.get("AZURE_FACE_ENDPOINT", "").strip()
    return bool(key and ep)


def _age_source() -> str:
    """qaytadi: 'onnx' | 'azure' | 'none' — auto rejimda ONNX mavjud bo'lsa u birinchi."""
    mode = (os.environ.get("AGE_BACKEND") or "auto").strip().lower()
    has_onnx = ONNX_PATH.is_file()
    has_azure = _azure_configured()
    if mode == "azure":
        if has_azure:
            return "azure"
        return "onnx" if has_onnx else "none"
    if mode == "onnx":
        return "onnx" if has_onnx else "none"
    # auto
    if has_onnx:
        return "onnx"
    if has_azure:
        return "azure"
    return "none"


def _azure_predict_age_bgr(
    bgr: np.ndarray,
) -> tuple[int | None, str, tuple[int, int, int, int] | None]:
    """To'liq kadr yuboriladi; Azure yuz + yosh qaytaradi."""
    key = os.environ.get("AZURE_FACE_KEY", "").strip()
    endpoint = os.environ.get("AZURE_FACE_ENDPOINT", "").strip().rstrip("/")
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None, "Rasmni JPEG qilishda xato.", None
    body = buf.tobytes()
    q = "returnFaceAttributes=age&detectionModel=detection_01"
    url = f"{endpoint}/face/v1.0/detect?{q}"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Ocp-Apim-Subscription-Key", key)
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            faces = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace") if e.fp else str(e)
        return None, f"Azure HTTP {e.code}: {msg[:500]}", None
    except urllib.error.URLError as e:
        return None, f"Tarmoq/Azure: {e.reason}", None

    if not faces:
        return None, "Azure: kadrda yuz aniqlanmadi.", None

    def _area(f: dict) -> int:
        r = f["faceRectangle"]
        return int(r["width"]) * int(r["height"])

    best = max(faces, key=_area)
    age_f = float(best["faceAttributes"]["age"])
    r = best["faceRectangle"]
    box = (int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"]))
    return int(round(age_f)), "", box


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def _get_yunet() -> Any:
    """YuNet detektori; fayl bo'lmasa bir marta yuklab olishga urinadi, bo'lmasa None."""
    global _yunet, _yunet_failed
    if _yunet is not None or _yunet_failed:
        return _yunet
    try:
        if not YUNET_PATH.is_file():
            YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        _yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), YUNET_SCORE, 0.3, 5000)
    except Exception as e:  # noqa: BLE001
        print(f"YuNet ishlamadi ({e}) — Haar kaskadi ishlatiladi.")
        _yunet_failed = True
    return _yunet


def _get_onnx_session() -> tuple[ort.InferenceSession, str]:
    global _ort_session, _ort_input_name, _img_size
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(
            f"ONNX topilmadi: {ONNX_PATH}\n"
            "Azure sozlamagan bo'lsangiz, train_age.py bilan age_predictor.onnx yarating."
        )
    if _ort_session is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        nthr = max(1, min(8, os.cpu_count() or 4))
        so.intra_op_num_threads = nthr
        so.inter_op_num_threads = 1
        _ort_session = ort.InferenceSession(
            str(ONNX_PATH),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        inp = _ort_session.get_inputs()[0]
        _ort_input_name = inp.name
        # Kirish o'lchami modeldan olinadi (yangi model 160, eski 224)
        if len(inp.shape) == 4 and isinstance(inp.shape[2], int):
            _img_size = int(inp.shape[2])
    assert _ort_input_name is not None
    return _ort_session, _ort_input_name


def _model_info() -> str:
    """ONNX metadata (train_age.py yozadi): arxitektura, kirish o'lchami, val MAE."""
    try:
        sess, _ = _get_onnx_session()
    except FileNotFoundError:
        return "model yo'q"
    meta = sess.get_modelmeta().custom_metadata_map
    if meta.get("arch") == "resnet18_dldl":
        return (
            f"ResNet18 + DLDL ({meta.get('dataset', 'UTKFace')}, {meta.get('img_size', _img_size)}px), "
            f"val MAE {meta.get('val_mae', '?')} yil"
        )
    return f"eski model ({_img_size}px, metadata yo'q) — train_age.py bilan qayta o'qiting"


def _crop_like_training(
    rgb: np.ndarray, box: tuple[int, int, int, int], params: tuple[float, float]
) -> np.ndarray:
    """Detektor qutisini UTKFace uslubidagi kvadrat qirqimga aylantirish (chetda — to'ldirish)."""
    x, y, w, h = box
    scale, shift_y = params
    side = max(2, int(round(scale * w)))
    cx = x + w / 2.0
    cy = y + h / 2.0 + shift_y * h
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    H, W = rgb.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - W), max(0, y0 + side - H)
    crop = rgb[max(0, y0) : min(H, y0 + side), max(0, x0) : min(W, x0 + side)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return crop


def _preprocess_face_rgb(face_rgb: np.ndarray, size: int) -> np.ndarray:
    if face_rgb.size == 0:
        raise ValueError("Bo'sh yuz qirqindisi")
    h, w = face_rgb.shape[:2]
    interp = cv2.INTER_AREA if h >= size and w >= size else cv2.INTER_LINEAR
    resized = cv2.resize(face_rgb, (size, size), interpolation=interp)
    t = resized.astype(np.float32) / 255.0
    t = (t - _MEAN) / _STD
    return np.transpose(t, (2, 0, 1))


def _onnx_predict_age(face_rgb: np.ndarray) -> float:
    """Asl + ko'zgu (flip) tasvir bo'yicha o'rtacha — bitta tasvirdan barqarorroq."""
    sess, name = _get_onnx_session()
    t = _preprocess_face_rgb(face_rgb, _img_size)
    batch = np.ascontiguousarray(np.stack([t, t[:, :, ::-1]]), dtype=np.float32)
    try:
        out = sess.run(None, {name: batch})[0]
    except Exception:  # noqa: BLE001 — eski ONNX batch=1 bilan qotgan bo'lishi mumkin
        out = np.concatenate([sess.run(None, {name: b[None]})[0] for b in batch])
    age = float(np.asarray(out).reshape(-1).mean())
    return max(0.0, min(120.0, age))


def _detect_faces_yunet(det: Any, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    s = min(1.0, YUNET_MAX_SIDE / float(max(h, w)))
    work = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s)))) if s < 1.0 else bgr
    det.setInputSize((work.shape[1], work.shape[0]))
    _, faces = det.detect(work)
    if faces is None:
        return []
    boxes = [tuple(int(round(v / s)) for v in f[:4]) for f in faces]
    return sorted(boxes, key=lambda b: -b[2] * b[3])


def _detect_faces(bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], tuple[float, float]]:
    """Barcha yuzlar (kattasidan kichigiga) + shu detektor uchun qirqish parametrlari."""
    det = _get_yunet()
    if det is not None:
        try:
            return _detect_faces_yunet(det, bgr), CROP_PARAMS["yunet"]
        except cv2.error:
            pass
    return _detect_faces_haar(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)), CROP_PARAMS["haar"]


def _detect_faces_haar(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Haar: barcha yuzlar (kattasidan kichigiga). Ba'zi o'lchamlarda assert xatosidan himoya."""
    if gray is None or gray.ndim != 2:
        return []
    h, w = int(gray.shape[0]), int(gray.shape[1])
    if h < 2 or w < 2:
        return []

    # Juda kichik kadrda piramida buziladi — biroz kattalashtirib deteksiya
    scale_back = 1.0
    work = gray
    min_side = min(h, w)
    if min_side < 120:
        scale_back = 240.0 / float(min_side)
        nw = max(2, int(round(w * scale_back)))
        nh = max(2, int(round(h * scale_back)))
        work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)

    wh, ww = int(work.shape[0]), int(work.shape[1])
    min_w = max(8, min(40, ww // 5, max(1, ww - 1)))
    min_h = max(8, min(40, wh // 5, max(1, wh - 1)))

    # Yorug'lik farqlarini tekislash — qorong'i kamerada deteksiya yaxshilanadi
    work = cv2.equalizeHist(work)
    cascade = _get_face_cascade()
    faces: Any = ()
    # Avval qat'iy (kam yolg'on yuz), topilmasa — yumshoqroq
    for neighbors in (5, 3):
        try:
            faces = cascade.detectMultiScale(
                work,
                scaleFactor=1.1,
                minNeighbors=neighbors,
                minSize=(min_w, min_h),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error:
            return []
        if len(faces):
            break

    out: list[tuple[int, int, int, int]] = []
    for r in sorted(faces, key=lambda r: -int(r[2]) * int(r[3])):
        x, y, fw, fh = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        if scale_back != 1.0:
            x = max(0, min(int(round(x / scale_back)), w - 1))
            y = max(0, min(int(round(y / scale_back)), h - 1))
            fw = min(max(1, int(round(fw / scale_back))), w - x)
            fh = min(max(1, int(round(fh / scale_back))), h - y)
        out.append((x, y, fw, fh))
    return out


def _ensure_bgr(image: np.ndarray) -> np.ndarray | None:
    """Gradio RGB / kulrang / RGBA — BGR 3 kanal."""
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        return None
    ch = image.shape[2]
    if ch == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if ch == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return None


Box = tuple[int, int, int, int]


def _predict_faces_bgr(
    bgr: np.ndarray, max_faces: int = 1
) -> tuple[list[tuple[float, Box]], str, Box | None]:
    """Qaytadi: [(yosh, quti), ...] (kattasidan), xato matni, xato bo'lsa — tegishli quti."""
    src = _age_source()
    if src == "azure":
        age, detail, box = _azure_predict_age_bgr(bgr)
        return ([(float(age), box)] if age is not None and box else []), detail, box
    if src == "none":
        return [], "ONNX yo'q va Azure ham sozlanmagan. train_age.py yoki AGE_BACKEND/Azure.", None

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.shape[0] < 2 or bgr.shape[1] < 2:
        return [], "Rasm o'lchami noto'g'ri yoki juda kichik.", None

    faces, crop_params = _detect_faces(bgr)
    if not faces:
        return [], "Yuz topilmadi — yuzni yaxshiroq ko'ring.", None
    if faces[0][2] < 40 or faces[0][3] < 40:
        return [], "Yuz juda kichik — kameraga yaqinroq keling.", faces[0]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results: list[tuple[float, Box]] = []
    for box in faces[:max_faces]:
        if box[2] < 40 or box[3] < 40:
            break
        try:
            results.append((_onnx_predict_age(_crop_like_training(rgb, box, crop_params)), box))
        except FileNotFoundError as e:
            return [], str(e), box
        except Exception as e:  # noqa: BLE001
            return [], f"Model xatosi: {e}", box
    return results, "", None


def _predict_age_bgr(bgr: np.ndarray) -> tuple[float | None, str, Box | None]:
    """Eng katta yuz uchun yosh (kamera rejimi)."""
    results, detail, err_box = _predict_faces_bgr(bgr, max_faces=1)
    if not results:
        return None, detail, err_box
    age, box = results[0]
    return age, "", box


def _backend_label() -> str:
    m = {"onnx": "lokal ONNX (tez)", "azure": "Azure Face API", "none": "model yo'q"}
    return m.get(_age_source(), "noma'lum")


def _draw_overlay_rgb(
    frame_rgb: np.ndarray,
    text: str,
    box: tuple[int, int, int, int] | None,
    *,
    color_bgr: tuple[int, int, int] = (60, 220, 110),
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, 2)
    cv2.putText(
        bgr,
        text,
        (16, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process_webcam_frame(
    frame: np.ndarray | None, st: dict[str, Any] | None
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    if st is None:
        st = _new_stream_state()
    if frame is None:
        return None, "Kamera kutilmoqda...", st

    now = time.monotonic()
    if now - st["last_analyze"] >= STREAM_ANALYZE_INTERVAL_SEC:
        st["last_analyze"] = now
        bgr = _ensure_bgr(frame)
        if bgr is None:
            st["age"] = None
            st["detail"] = "Kadr formati noto'g'ri."
            st["box"] = None
        else:
            try:
                age, detail, box = _predict_age_bgr(bgr)
            except Exception as e:  # noqa: BLE001
                age, detail, box = None, f"Xatolik: {e}", None
            if age is not None:
                # Ketma-ket kadrlar bo'yicha silliqlash — raqam sakramaydi
                ema = st["age_ema"]
                if ema is None or now - st["last_face"] > STREAM_RESET_AFTER_SEC:
                    ema = age
                else:
                    ema = STREAM_EMA_ALPHA * age + (1 - STREAM_EMA_ALPHA) * ema
                st["age_ema"] = ema
                st["last_face"] = now
                st["age"] = int(round(ema))
            else:
                st["age"] = None
            st["detail"] = detail
            st["box"] = box

    age = st["age"]
    detail = st["detail"]
    box = st["box"]
    src = _backend_label()

    if age is not None:
        label = f"Taxminiy yosh: {age}"
        readout = f"Taxminiy yosh: {age} ({src})"
    elif detail:
        label = detail
        readout = detail
    else:
        label = "Yuz qidirilmoqda..."
        readout = "Yuz qidirilmoqda..."

    return _draw_overlay_rgb(frame, label, box), readout, st


def process_upload(image: np.ndarray | None) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Rasm yuklang."

    bgr = _ensure_bgr(image)
    if bgr is None:
        return None, "Rasm formati qo'llab-quvvatlanmaydi (RGB/kulrang/RGBA kutiladi)."

    try:
        results, detail, _ = _predict_faces_bgr(bgr, max_faces=10)
    except Exception as e:  # noqa: BLE001
        return Image.fromarray(image), f"Xatolik: {e}"

    pil = Image.fromarray(image).convert("RGB")
    if not results:
        return pil, detail or "Yuz topilmadi."

    draw = ImageDraw.Draw(pil)
    color = (60, 220, 110)
    lw = max(2, round(min(pil.size) / 250))
    lines = []
    # Chapdan o'ngga raqamlash — matndagi tartib rasmdagi bilan mos bo'lsin
    results.sort(key=lambda r: r[1][0])
    for i, (age_f, (x, y, w, h)) in enumerate(results, 1):
        age = int(round(age_f))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        font = ImageFont.load_default(size=max(14, int(h * 0.22)))
        label = f"#{i}: {age}" if len(results) > 1 else f"Yosh: {age}"
        tb = draw.textbbox((0, 0), label, font=font)
        ty = y - (tb[3] - tb[1]) - 2 * lw if y > (tb[3] - tb[1]) + 2 * lw else y + h + lw
        draw.rectangle([x, ty, x + tb[2] + 2 * lw, ty + tb[3] + lw], fill=color)
        draw.text((x + lw, ty), label, fill=(0, 0, 0), font=font)
        lines.append(f"{i}-yuz: {age} yosh" if len(results) > 1 else f"Natija: {age} yosh.")
    src = _backend_label()
    return pil, "\n".join(lines) + f"\nManba: {src}."


def reset_stream_cache() -> tuple[str, dict[str, Any]]:
    return "Holat yangilandi. Kamerani qayta yoqing.", _new_stream_state()


def _startup_help_md() -> str:
    src = _age_source()
    onnx_ok = ONNX_PATH.is_file()
    model = _model_info() if src == "onnx" else _backend_label()
    parts = [
        "## Yosh bashorati (real vaqt + rasm)\n",
        "### Aniqlik haqida\n"
        f"Chiqadigan yosh **taxminiy**dir. Model: **{model}**. MAE — o‘rtacha mutlaq xato: "
        "bashorat haqiqiy yoshdan o‘rtacha shuncha yilga farq qiladi. Yaxshi natija uchun: "
        "yuz **old tomondan**, yaxshi **yoritilgan**, ko‘zoynak/niqobsiz va kadrda yetarlicha "
        "**katta** bo‘lsin. Kamerada natija bir necha kadr bo‘yicha silliqlanadi.\n\n",
        "### Tez lokal (standart)\n",
        "**ONNX** `checkpoints/age_predictor.onnx` bo'lsa — **avvalo u** ishlatiladi "
        "(internet kerak emas, kamera uchun eng tez).\n\n",
        f"**Faol manba:** `{src}`\n\n",
    ]
    if onnx_ok:
        parts.append(f"ONNX mavjud: `{ONNX_PATH}`\n\n")
    else:
        parts.append(f"ONNX yo'q: `{ONNX_PATH}` — `train_age.py` bilan yarating.\n\n")
    if _azure_configured():
        parts.append(
            "Azure kalitlari ham bor; standartda ONNX ustun. "
            "Faqat Azure: `$env:AGE_BACKEND='azure'`\n\n"
        )
    else:
        parts.append("Azure (ixtiyoriy): `AGE_BACKEND=azure` + kalitlar.\n\n")
    parts.append(
        "### Azure namunasi\n```\n"
        "$env:AZURE_FACE_KEY=\"...\"\n"
        "$env:AZURE_FACE_ENDPOINT=\"https://....cognitiveservices.azure.com\"\n"
        "$env:AGE_BACKEND=\"azure\"\n"
        "python app.py\n```"
    )
    return "".join(parts)


with gr.Blocks(title="Yosh bashorati — yuz bo'yicha") as demo:
    gr.Markdown(_startup_help_md)

    with gr.Tab("Real vaqt kamera"):
        gr.Markdown(
            "**Natija** o‘ngda. ONNX ham Azure ham yo‘q bo‘lsa natija bo‘sh qoladi."
        )
        age_readout = gr.Textbox(
            label="Taxminiy yosh (matn)",
            interactive=False,
            lines=3,
            value="Kamerani yoqing...",
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Jonli kamera**")
                cam_in = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    label="Kamera",
                    streaming=True,
                )
            with gr.Column():
                gr.Markdown("**Natija**")
                out_cam = gr.Image(label="Natija (yosh + ramka)", type="numpy")
        reset_btn = gr.Button("Keshni tozalash")
        reset_msg = gr.Textbox(label="Holat", interactive=False)
        cam_state = gr.State(None)

        cam_in.stream(
            fn=process_webcam_frame,
            inputs=[cam_in, cam_state],
            outputs=[out_cam, age_readout, cam_state],
            stream_every=0.05,
        )
        reset_btn.click(fn=reset_stream_cache, outputs=[reset_msg, cam_state])

    with gr.Tab("Rasm yuklash"):
        up = gr.Image(type="numpy", label="Rasm", sources=["upload"])
        up_out = gr.Image(type="pil", label="Natija")
        up_txt = gr.Textbox(label="Matn", interactive=False)

        def _on_upload(img: np.ndarray | None):
            return process_upload(img)

        up.upload(fn=_on_upload, inputs=up, outputs=[up_out, up_txt])
        gr.Markdown("Rasmda bir nechta yuz bo‘lsa — har biri alohida belgilanadi (10 tagacha).")

    gr.Markdown(
        "**O‘qitish (ixtiyoriy, ONNX uchun):** `python train_age.py --data_dir \"...\"`"
    )


if __name__ == "__main__":
    src = _age_source()
    print(f"Bashorat manbasi: {src}")
    if src == "onnx":
        print(f"  ONNX: {ONNX_PATH}")
        print(f"  Model: {_model_info()}")
        print(f"  Yuz detektori: {'YuNet' if _get_yunet() is not None else 'Haar'}")
    elif src == "azure":
        print("  Azure Face API")
    else:
        print(
            "  Ogohlantirish: model yo'q — train_age.py (ONNX) yoki Azure sozlang."
        )
    demo.queue()
    demo.launch()
