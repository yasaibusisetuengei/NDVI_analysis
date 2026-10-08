import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def calculate_ndvi(image_array):
    # OpenCVを用いてBGRチャンネルに分解
    # 一般的な改造カメラの仕様として、RチャンネルにNIR、Bチャンネルに可視光が割り当てられていると仮定
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
    # 植生状態の可視化に一般的なJETカラーマップを使用
    ndvi_color = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    
    # Streamlit表示用にBGRからRGBへ変換
    return cv2.cvtColor(ndvi_color, cv2.COLOR_BGR2RGB)

st.title("NDVI 解析 & 相対応答グラフビューア")
st.write("画像をアップロードすると、NDVI画像とカメラの相対応答グラフを生成します。")

uploaded_file = st.file_uploader("対象の画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # OpenCV処理用にRGBからBGRへ一時変換
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.subheader("1. NDVI 画像の生成")
    # NDVI計算とカラーマップ適用
    ndvi_data = calculate_ndvi(img_bgr)
    ndvi_colored = create_ndvi_colormap(ndvi_data)
    
    # 左右に並べて表示
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="オリジナル画像", use_column_width=True)
    with col2:
        st.image(ndvi_colored, caption="NDVI 疑似カラー画像", use_column_width=True)


    st.subheader("2. 相対応答 (Relative Response) グラフ")
    # メーカー仕様書に基づく波長と相対応答のデータを定義
    # ※以下はデュアルバンドパスフィルターの一般的な特性を模したダミーデータです。
    # 実際のカメラ仕様（Bizworks社のデータ）の数値に書き換えてください。
    
    wavelength = np.linspace(0.4, 1.0, 100) # 波長 0.4 um から 1.0 um
    # 可視光(青)領域と近赤外領域のピークをシミュレート
    response_vis = np.exp(-((wavelength - 0.45) / 0.05)**2) 
    response_nir = np.exp(-((wavelength - 0.82) / 0.08)**2) 
    relative_response = np.maximum(response_vis, response_nir) * 100 # 0-100%スケール

    # グラフの描画
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelength, relative_response, color="purple", linewidth=2)
    
    # 指定された軸ラベルを設定
    ax.set_xlabel("wavelength [micrometer]")
    ax.set_ylabel("relative response")
    
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    st.pyplot(fig)
