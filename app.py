"""
Yuz tasviri bo'yicha yosh bashorati: real vaqt + rasm yuklash.

Tez lokal (standart): checkpoints/age_predictor.onnx — birinchi navbatda ishlatiladi.

Ixtiyoriy bulut: AGE_BACKEND=azure + AZURE_FACE_KEY / AZURE_FACE_ENDPOINT.

Yuz: ONNX rejimida OpenCV Haar; Azure da — API ichki deteksiya.
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
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "checkpoints" / "age_predictor.onnx"

_IMG_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Kichikroq = tezroq yangilanish (CPU yuklari ortishi mumkin)
STREAM_ANALYZE_INTERVAL_SEC = 0.5

_stream_state: dict[str, Any] = {
    "last_analyze": 0.0,
    "age": None,
    "detail": "",
    "box": None,
}

_face_cascade: cv2.CascadeClassifier | None = None
_ort_session: ort.InferenceSession | None = None
_ort_input_name: str | None = None


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


def _get_onnx_session() -> tuple[ort.InferenceSession, str]:
    global _ort_session, _ort_input_name
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
        _ort_input_name = _ort_session.get_inputs()[0].name
    assert _ort_input_name is not None
    return _ort_session, _ort_input_name


def _preprocess_face_rgb(face_rgb: np.ndarray) -> np.ndarray:
    if face_rgb.size == 0:
        raise ValueError("Bo'sh yuz qirqindisi")
    # Kichraytirishda INTER_AREA odatda tezroq/legacy CPU uchun mos
    h, w = face_rgb.shape[:2]
    interp = cv2.INTER_AREA if h >= _IMG_SIZE and w >= _IMG_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(face_rgb, (_IMG_SIZE, _IMG_SIZE), interpolation=interp)
    t = resized.astype(np.float32) / 255.0
    t = (t - _MEAN) / _STD
    t = np.transpose(t, (2, 0, 1))
    return np.expand_dims(t, axis=0)


def _onnx_predict_age(face_rgb: np.ndarray) -> float:
    sess, name = _get_onnx_session()
    inp = _preprocess_face_rgb(face_rgb)
    out = sess.run(None, {name: inp})[0]
    age = float(np.asarray(out).reshape(-1)[0])
    return max(0.0, min(120.0, age))


def _largest_face(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=4,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda r: int(r[2]) * int(r[3]))


def _predict_age_bgr(bgr: np.ndarray) -> tuple[int | None, str, tuple[int, int, int, int] | None]:
    src = _age_source()
    if src == "azure":
        return _azure_predict_age_bgr(bgr)
    if src == "none":
        return (
            None,
            "ONNX yo'q va Azure ham sozlanmagan. train_age.py yoki AGE_BACKEND/Azure.",
            None,
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face = _largest_face(gray)
    if face is None:
        return None, "Yuz topilmadi — yuzni yaxshiroq ko'ring.", None

    x, y, w, h = [int(v) for v in face]
    if w < 40 or h < 40:
        return None, "Yuz juda kichik.", (x, y, w, h)

    face_bgr = bgr[y : y + h, x : x + w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    try:
        age_f = _onnx_predict_age(face_rgb)
    except FileNotFoundError as e:
        return None, str(e), (x, y, w, h)
    except Exception as e:  # noqa: BLE001
        return None, f"Model xatosi: {e}", (x, y, w, h)

    return int(round(age_f)), "", (x, y, w, h)


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


def process_webcam_frame(frame: np.ndarray | None) -> tuple[np.ndarray | None, str]:
    if frame is None:
        return None, "Kamera kutilmoqda..."

    now = time.monotonic()
    if now - _stream_state["last_analyze"] >= STREAM_ANALYZE_INTERVAL_SEC:
        _stream_state["last_analyze"] = now
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            age, detail, box = _predict_age_bgr(bgr)
            _stream_state["age"] = age
            _stream_state["detail"] = detail
            _stream_state["box"] = box
        except Exception as e:  # noqa: BLE001
            _stream_state["age"] = None
            _stream_state["detail"] = f"Xatolik: {e}"
            _stream_state["box"] = None

    age = _stream_state["age"]
    detail = _stream_state["detail"]
    box = _stream_state["box"]
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

    return _draw_overlay_rgb(frame, label, box), readout


def process_upload(image: np.ndarray | None) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Rasm yuklang."

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        age, detail, box = _predict_age_bgr(bgr)
    except Exception as e:  # noqa: BLE001
        return Image.fromarray(image), f"Xatolik: {e}"

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    if age is None:
        return pil, detail or "Yuz topilmadi."

    if box:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline=(60, 220, 110), width=3)
    text = f"Taxminiy yosh: {age}"
    draw.text((22, 18), text, fill=(60, 220, 110))
    src = _backend_label()
    return pil, f"Natija: {age} yosh. Manba: {src}."


def reset_stream_cache() -> str:
    _stream_state["last_analyze"] = 0.0
    _stream_state["age"] = None
    _stream_state["detail"] = ""
    _stream_state["box"] = None
    return "Holat yangilandi. Kamerani qayta yoqing."


def _startup_help_md() -> str:
    src = _age_source()
    onnx_ok = ONNX_PATH.is_file()
    parts = [
        "## Yosh bashorati (real vaqt + rasm)\n",
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

        cam_in.stream(
            fn=process_webcam_frame,
            inputs=[cam_in],
            outputs=[out_cam, age_readout],
            stream_every=0.05,
        )
        reset_btn.click(fn=reset_stream_cache, outputs=reset_msg)

    with gr.Tab("Rasm yuklash"):
        up = gr.Image(type="numpy", label="Rasm", sources=["upload"])
        up_out = gr.Image(type="pil", label="Natija")
        up_txt = gr.Textbox(label="Matn", interactive=False)

        def _on_upload(img: np.ndarray | None):
            return process_upload(img)

        up.upload(fn=_on_upload, inputs=up, outputs=[up_out, up_txt])

    gr.Markdown(
        "**O‘qitish (ixtiyoriy, ONNX uchun):** `python train_age.py --data_dir \"...\"`"
    )


if __name__ == "__main__":
    src = _age_source()
    print(f"Bashorat manbasi: {src}")
    if src == "onnx":
        print(f"  ONNX: {ONNX_PATH}")
        _get_onnx_session()
    elif src == "azure":
        print("  Azure Face API")
    else:
        print(
            "  Ogohlantirish: model yo'q — train_age.py (ONNX) yoki Azure sozlang."
        )
    demo.queue()
    demo.launch()
