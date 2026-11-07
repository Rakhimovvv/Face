# Yosh bashorati (yuz rasmi)

Yuz surati asosida **taxminiy yosh** aniqlaydigan loyiha: **jonli kamera** va **rasm yuklash** (Gradio). Model **UTKFace** to‘plamida o‘qitiladi, natijada **ONNX** fayl chiqadi va ilova shu fayl bilan internetsiz ishlaydi.

## Qanday ishlaydi

1. **Yuz aniqlash** — OpenCV **YuNet** (DNN detektor; fayl yo‘q bo‘lsa bir marta yuklab olinadi, ishlamasa **Haar** kaskadiga qaytadi).
2. **Qirqish** — detektor qutisi UTKFace “aligned & cropped” uslubidagi kvadratga aylantiriladi (`CROP_PARAMS`, har bir detektor uchun UTKFace rasmlarida o‘lchangan). Model qanday qirqimda o‘qitilgan bo‘lsa, ilovada ham shunday qirqim berilishi aniqlik uchun muhim.
3. **Model** — ImageNet’da oldindan o‘qitilgan **ResNet18** (transfer learning) + 0..100 yosh uchun 101 sinfli bosh. O‘qitish **DLDL** usulida: har bir yosh Gauss yorliq taqsimotiga (σ = 2 yil) aylantiriladi, yo‘qotish = KL-divergensiya + 0.1 · L1. Chiqish — kutilgan yosh (ehtimolliklar bo‘yicha o‘rtacha).
4. **TTA** — asl va ko‘zgu (gorizontal flip) tasvir bashoratlarining o‘rtachasi.
5. **Kamera** — natija ketma-ket kadrlar bo‘yicha eksponensial silliqlanadi (α = 0.35); yuz 1.5 s ko‘rinmasa silliqlash qaytadan boshlanadi. Holat har bir brauzer sessiyasi uchun alohida.

## Texnologiyalar

- **Python**, **OpenCV** (YuNet / Haar)
- **PyTorch**, **torchvision** (`train_age.py` — o‘qitish)
- **ONNX Runtime** (`app.py` — tez lokal bashorat)
- **Gradio** (veb-interfeys)
- Ixtiyoriy: **Azure Face API** (`AGE_BACKEND=azure`)

## O‘rnatish

```powershell
cd "C:\Users\ulugb\Desktop\Dilom ishi"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# O‘qitish uchun qo‘shimcha:
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

> Windowsda tizimdagi `python` boshqa interpretator bo‘lishi mumkin — har doim **`.venv\Scripts\python.exe`** ni ishlating (PyTorch shu yerda o‘rnatilgan).

## Ma’lumotlar (UTKFace)

23 708 ta yuz rasmi, fayl nomi `[yosh]_[jins]_[irq]_[sana].jpg.chip.jpg`. “Aligned & cropped” varianti HuggingFace’da mavjud:

```powershell
curl.exe -L -o UTKFace.tar.gz https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/main/UTKFace.tar.gz
mkdir data
tar -xzf UTKFace.tar.gz -C data      # natija: data\UTKFace\*.jpg
del UTKFace.tar.gz
```

`data/` papkasi `.gitignore` da — repozitoriyga tushmaydi.

## Modelni o‘qitish

```powershell
.\.venv\Scripts\python.exe train_age.py --data_dir data\UTKFace --epochs 8
```

- Ma’lumotlar `--seed 42` bilan aralashtiriladi: 90% train (21 338), 10% val (2 370).
- CPU’da (Intel i5-12450H) bir epoxa ~11 daqiqa, 8 epoxa ≈ 1.5 soat; kompyuter boshqa ish bilan band bo‘lsa 2 baravargacha sekinlashadi. GPU bo‘lsa avtomatik ishlatiladi.
- **RAM kam bo‘lsa (8 GB)** — o‘qitish paytida brauzerdagi kamera va og‘ir dasturlarni yopib turing; jarayon yiqilsa `--workers 1` yoki `--workers 0` bilan qayta urinib ko‘ring (Windows’da har bir DataLoader jarayoni torch’ni qayta yuklab, xotirani band qiladi).
- **Uzilib qolsa** — xuddi shu argumentlar bilan `--resume` qo‘shing, oxirgi tugagan epoxadan davom etadi:

  ```powershell
  .\.venv\Scripts\python.exe train_age.py --data_dir data\UTKFace --epochs 8 --resume
  ```

- **Faqat ONNX qayta chiqarish** (`best.pth` bor bo‘lsa): `train_age.py --export_only`
- Tez sinov uchun: `--max_samples 500 --epochs 1`

`checkpoints/` ga yoziladigan fayllar:

| Fayl | Tavsif |
|------|--------|
| `age_predictor.onnx` | Ilova ishlatadigan model (ichida metadata: arxitektura, o‘lcham, val MAE) |
| `best.pth` | Eng yaxshi val MAE bo‘lgan epoxa vaznlari |
| `last.pth` | Oxirgi epoxa holati (optimizator, scheduler) — `--resume` uchun |
| `train_log.txt` | To‘liq o‘qitish logi, oxirida yosh guruhlari bo‘yicha MAE |
| `history.csv` | Har epoxa: `train_mae`, `val_mae`, vaqt — grafik chizish uchun |

## Natijalar

MAE — o‘rtacha mutlaq xato (yil): bashorat haqiqiy yoshdan o‘rtacha shuncha farq qiladi. Bir xil val to‘plamida (2 370 rasm, flip TTA bilan):

| Model | Kirish | Val MAE | ±5 yil ichida |
|-------|--------|---------|---------------|
| Eski oddiy CNN (4 qatlam, L1 regressiya) | 224 px | 15.05 | 30.1% |
| **ResNet18 + DLDL** (8 epoxa, CPU) | 160 px | **4.44** | **68.0%** |

Yosh guruhlari bo‘yicha MAE (yil):

| Guruh | Rasmlar | Eski CNN | ResNet18 + DLDL |
|-------|---------|----------|-----------------|
| 0–12 | 351 | 27.12 | **1.57** |
| 13–19 | 122 | 14.59 | **4.32** |
| 20–29 | 727 | 4.99 | **3.30** |
| 30–39 | 460 | 4.63 | **4.17** |
| 40–49 | 225 | 12.85 | **5.94** |
| 50–59 | 216 | 22.82 | **7.70** |
| 60–100 | 269 | 40.10 | **7.93** |

Eski model deyarli hammaga 25–35 atrofida javob bergan, shuning uchun faqat 20–39 yoshlarda yaxshi ko‘ringan. Yangi model barcha guruhlarda aniqroq. Xato yosh oshgan sari o‘sadi: keksalar ma’lumotda kam va bir xil yoshdagi odamlarning tashqi ko‘rinishi ham ko‘proq farq qiladi.

O‘qitish jarayoni (`checkpoints/history.csv`):

| Epoxa | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|---|---|---|---|---|---|---|---|
| Train MAE | 9.22 | 5.92 | 5.32 | 4.96 | 4.59 | 4.28 | 4.00 | 3.86 |
| Val MAE | 6.97 | 5.40 | 5.13 | 4.76 | 4.69 | 4.51 | 4.45 | 4.44 |

> Eski model boshqa bo‘linmada o‘qitilgan, shuning uchun bu val rasmlarining bir qismini o‘qitishda ko‘rgan bo‘lishi mumkin. Bu unga ustunlik beradi, ya’ni haqiqiy farq jadvaldagidan ham katta bo‘lishi mumkin.

## Ilovani ishga tushirish

```powershell
.\.venv\Scripts\python.exe app.py
```

Brauzerda ochiladi (odatda http://127.0.0.1:7860). Sahifa yuqorisida faol model va uning val MAE ko‘rsatiladi.

### Muhit o‘zgaruvchilari (ixtiyoriy)

| O‘zgaruvchi | Vazifasi |
|-------------|----------|
| `AZURE_FACE_KEY`, `AZURE_FACE_ENDPOINT` | Azure Face API kalitlari |
| `AGE_BACKEND` | `auto` (standart: ONNX bo‘lsa — ONNX), `onnx` yoki `azure` |

## Loyiha tuzilishi

| Fayl | Tavsif |
|------|--------|
| `app.py` | Gradio: kamera + yuklash, YuNet/Haar, ONNX / Azure |
| `train_age.py` | ResNet18 + DLDL o‘qitish, `--resume`, `--export_only`, ONNX eksport |
| `requirements.txt` | Ilova bog‘liqliklari |
| `requirements-train.txt` | O‘qitish (PyTorch, onnx, …) |

Katta fayllar (`.venv`, `checkpoints/`, `models/`, `data/`, `*.onnx`, `*.pth`) `.gitignore` orqali repozitoriyga kiritilmagan — boshqa kompyuterda yuqoridagi qadamlar bilan qayta yaratiladi.

## Cheklovlar

- Chiqqan raqam — **statistik taxmin**, pasport yoshi emas. Bitta odam uchun xato o‘rtacha MAE dan katta bo‘lishi ham mumkin.
- UTKFace’da bolalar va keksalar kamroq — bu guruhlarda xato kattaroq (natijalar jadvaliga qarang).
- Yorug‘lik, bosh burilishi, ko‘zoynak/niqob, kamera sifati natijaga ta’sir qiladi.

## Litsenziya va ma’lumotlar

UTKFace faqat **notijorat tadqiqot** maqsadida beriladi (Zhang, Song, Qi — *Age Progression/Regression by Conditional Adversarial Autoencoder*, CVPR 2017). Manbani ko‘rsating va foydalanish shartlariga rioya qiling.
