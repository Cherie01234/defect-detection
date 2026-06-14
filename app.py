"""Surface inspection demo — upload an image, get an anomaly score + heatmap.

    python data/make_synthetic.py   # once
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src import dataset, visualize
from src.detector import AnomalyDetector

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic"

st.set_page_config(page_title="外観検査 / Anomaly Detection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_detector():
    train = dataset.list_train_good(DATA)
    test_good = [s.path for s in dataset.list_test(DATA) if s.label == 0]
    if not train:
        return None
    det = AnomalyDetector().fit([dataset.load_image(p) for p in train])
    if test_good:
        det.calibrate([dataset.load_image(p) for p in test_good])
    return det


st.title("🔍 外観検査 — 正常品だけで学ぶ異常検知")
st.caption("正常画像のパッチ特徴メモリバンクと最近傍距離で、欠陥を検出し位置をヒートマップ表示します（PatchCore方式）。")

detector = load_detector()
if detector is None:
    st.error("合成データがありません。先に `python data/make_synthetic.py` を実行してください。")
    st.stop()

with st.sidebar:
    st.header("サンプル")
    st.caption("手元に画像が無ければ、テスト画像で試せます。")
    samples = dataset.list_test(DATA)
    pick = st.selectbox(
        "テスト画像を選択",
        options=samples,
        format_func=lambda s: f"{'欠陥' if s.label else '正常'} / {s.path.name}",
    )

uploaded = st.file_uploader("検査する画像をアップロード", type=["png", "jpg", "jpeg", "bmp"])
image = None
if uploaded is not None:
    from PIL import Image
    image = Image.open(uploaded).convert("RGB")
elif pick is not None:
    image = dataset.load_image(pick.path)

if image is not None and st.button("検査する", type="primary"):
    with st.spinner("特徴抽出 → 最近傍距離 → 異常マップ生成中..."):
        score, amap = detector.predict(image, out_size=256)

    verdict = "異常（NG）" if (detector.threshold and score > detector.threshold) else "正常（OK）"
    c1, c2 = st.columns(2)
    c1.metric("異常スコア", f"{score:.3f}",
              help="正常パッチからの最大距離。大きいほど異常。")
    c2.metric("判定", verdict)
    if detector.threshold:
        st.caption(f"しきい値: {detector.threshold:.3f}（正常画像から自動キャリブレーション）")

    st.subheader("異常マップ（原画像 / ヒートマップ / 重ね合わせ）")
    st.image(visualize.triptych(image, amap), use_column_width=True)
