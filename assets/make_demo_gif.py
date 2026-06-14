"""Generate a stylized demo GIF of the inspection UI flow.

The chrome is a Pillow mockup, but the anomaly score and the heatmap triptych
are produced by the *real* detector, so the result shown is genuine.

    python assets/make_demo_gif.py  ->  assets/demo.gif (+ assets/_preview.png)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import dataset, visualize  # noqa: E402
from src.detector import AnomalyDetector  # noqa: E402

DATA = ROOT / "data" / "synthetic"
W, H = 820, 620
BG = (255, 255, 255)
INK = (33, 37, 41)
GRAY = (130, 138, 148)
BORDER = (208, 212, 218)
RED = (255, 75, 75)
GREEN = (34, 139, 84)
DANGER = (200, 50, 50)

JP_FONTS = ["C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/arial.ttf"]


def font(size):
    for p in JP_FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_SUB, F_BODY, F_SMALL, F_BIG = font(25), font(14), font(16), font(13), font(30)


def run_detector():
    train = dataset.list_train_good(DATA)
    if not train:
        raise SystemExit("Run `python data/make_synthetic.py` first.")
    det = AnomalyDetector().fit([dataset.load_image(p) for p in train])
    test_good = [s.path for s in dataset.list_test(DATA) if s.label == 0]
    det.calibrate([dataset.load_image(p) for p in test_good])

    defects = [s for s in dataset.list_test(DATA) if s.label == 1]
    best = None
    for s in defects:
        img = dataset.load_image(s.path)
        score, amap = det.predict(img, out_size=256)
        if best is None or score > best[0]:
            best = (score, img, amap)
    score, img, amap = best
    trip = visualize.triptych(img, amap)
    return score, det.threshold, img.resize((200, 200)), trip


SCORE, THRESH, THUMB, TRIP = run_detector()
TRIP_W = 760
TRIP_SCALED = TRIP.resize((TRIP_W, int(TRIP.height * TRIP_W / TRIP.width)))


def render(stage: str, dots: int = 0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((30, 22), "外観検査 — 異常検知", font=F_TITLE, fill=INK)
    d.text((30, 58), "正常品の画像だけで学習し、欠陥を検出してヒートマップで可視化（PatchCore方式）",
           font=F_SUB, fill=GRAY)
    d.line((30, 88, W - 30, 88), fill=BORDER, width=1)

    # Input image
    d.text((30, 104), "入力画像", font=F_SMALL, fill=GRAY)
    img.paste(THUMB, (30, 126))
    d.rectangle((30, 126, 230, 326), outline=BORDER, width=1)

    # Button (hidden once results are shown)
    if stage != "done":
        pressed = stage == "press"
        d.rounded_rectangle((262, 150, 402, 196), 8, fill=DANGER if pressed else RED)
        d.text((300, 163), "検査する", font=F_BODY, fill=(255, 255, 255))

    # Status
    if stage in ("press", "run"):
        d.text((262, 214), "特徴抽出 → 最近傍距離 → 異常マップ生成中" + "." * dots,
               font=F_SMALL, fill=GRAY)

    # Results
    if stage == "done":
        d.text((262, 120), "異常スコア", font=F_SMALL, fill=GRAY)
        d.text((262, 138), f"{SCORE:.3f}", font=F_BIG, fill=INK)
        d.text((420, 120), "判定", font=F_SMALL, fill=GRAY)
        d.text((420, 138), "異常（NG）", font=F_BIG, fill=DANGER)
        d.text((262, 188), f"しきい値 {THRESH:.3f}（正常画像から自動算出）", font=F_SMALL, fill=GRAY)

        d.text((30, 344), "異常マップ：　原画像　／　ヒートマップ　／　重ね合わせ", font=F_SMALL, fill=GRAY)
        img.paste(TRIP_SCALED, (30, 368))

    return img


def build():
    frames, durations = [], []

    def add(im, ms):
        frames.append(im)
        durations.append(ms)

    add(render("idle"), 1500)
    add(render("press"), 250)
    for dots in (1, 2, 3, 2):
        add(render("run", dots), 350)
    add(render("done"), 3600)

    out = ROOT / "assets" / "demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    frames[-1].save(ROOT / "assets" / "_preview.png")
    print(f"Wrote {out} ({len(frames)} frames, {out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
