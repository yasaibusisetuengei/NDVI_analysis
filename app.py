import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ページ設定をワイドモードに変更して見やすく
st.set_page_config(layout="wide")

def calculate_ndvi(image_array):
    # OpenCVを用いてBGRチャンネルに分解
    b, g, r = cv2.split(image_array)
    
    r_float = r.astype(float)
    b_float = b.astype(float)
    
    # ゼロ除算を回避しながら NDVI = (NIR - VIS) / (NIR + VIS) を計算
    denominator = r_float + b_float
    ndvi = np.divide(r_float - b_float, denominator, out=np.zeros_like(r_float), where=denominator != 0)
    
    return ndvi

def create_ndvi_colormap(ndvi):
    # NDVIの値域(-1.0 〜 1.0)を0〜255に正規化してカラーマップを適用
    ndvi_normalized = ((ndvi + 1.0) / 2.0 * 255).astype(np.uint8)
    ndvi_color = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    return cv2.cvtColor(ndvi_color, cv2.COLOR_BGR2RGB)

def extract_vegetation(img_array, ndvi, threshold):
    # NDVIが閾値以上のピクセルをTrueとするマスクを作成
    mask = ndvi >= threshold
    
    # 元画像（RGB）をコピーして抽出用画像を作成
    extracted_img = img_array.copy()
    
    # 背景（マスクがFalseの領域）を黒（または白など）で塗りつぶす
    extracted_img[~mask] = [0, 0, 0] # ここを[255, 255, 255]にすると背景白
    
    # ピクセル数から面積比率を計算（参考情報）
    total_pixels = mask.size
    vegetation_pixels = np.sum(mask)
    area_ratio = (vegetation_pixels / total_pixels) * 100
    
    return extracted_img, mask, area_ratio

st.title("NDVI 解析 & 植生領域抽出ビューア")
st.write("画像をアップロードし、閾値を調整して葉や作物の領域を抽出します。")

# サイドバーに設定項目を移動
with st.sidebar:
    st.header("設定")
    uploaded_file = st.file_uploader("対象の画像をアップロードしてください", type=["jpg", "png", "jpeg"])
    
    st.subheader("植生抽出パラメータ")
    # NDVIの一般的な閾値(0.2〜0.5程度が植物の目安)を設定するスライダー
    ndvi_threshold = st.slider(
        "NDVI 閾値 (Threshold)", 
        min_value=-1.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="この値以上のNDVIを持つ領域を『植物（葉など）』として抽出します。通常、健康な葉は0.2〜0.8程度です。"
    )

if uploaded_file is not None:
    # 画像の読み込みと変換
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 1. NDVI計算とカラーマップ適用
    ndvi_data = calculate_ndvi(img_bgr)
    ndvi_colored = create_ndvi_colormap(ndvi_data)
    
    # 2. 閾値に基づく植生領域の抽出
    extracted_img, mask, area_ratio = extract_vegetation(img_array, ndvi_data, ndvi_threshold)

    st.subheader("画像解析結果")
    
    # 3列に分けて表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="オリジナル画像", use_column_width=True)
    with col2:
        st.image(ndvi_colored, caption="NDVI 疑似カラー画像 (JET)", use_column_width=True)
    with col3:
        st.image(extracted_img, caption=f"抽出領域 (閾値 >= {ndvi_threshold:.2f})", use_column_width=True)
        # 抽出した面積の比率を指標として表示
        st.metric(label="抽出面積比率 (画面全体比)", value=f"{area_ratio:.1f} %")


    st.subheader("相対応答 (Relative Response) グラフ")
    
    wavelength = np.linspace(0.4, 1.0, 100) 
    response_vis = np.exp(-((wavelength - 0.45) / 0.05)**2) 
    response_nir = np.exp(-((wavelength - 0.82) / 0.08)**2) 
    relative_response = np.maximum(response_vis, response_nir) * 100 

    fig, ax = plt.subplots(figsize=(10, 3)) # 高さを少し抑えて見やすく
    ax.plot(wavelength, relative_response, color="purple", linewidth=2)
    ax.set_xlabel("wavelength [micrometer]")
    ax.set_ylabel("relative response")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    st.pyplot(fig)
else:
    st.info("サイドバーから画像をアップロードしてください。")
