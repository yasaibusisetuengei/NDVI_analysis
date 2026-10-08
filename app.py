import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

# ページ基本設定
st.set_page_config(
    page_title="Yubaflex 分光感度 & NDVI シミュレーター",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Yubaflex 分光感度特性 & NDVI/フォールスカラー解析")
st.caption("参照仕様: BIZWORKS Yubaflex (https://www.bizworks.co.jp/YF/Spec.html)")

# ==========================================
# サイドバーコントロール
# ==========================================
st.sidebar.header("⚙️ ソフトウェア処理パラメータ")

# アルゴリズム調整用パラメータ
alpha = st.sidebar.slider(
    "赤(R)チャネルの近赤外減算係数 (α)",
    min_value=0.0, max_value=0.5, value=0.15, step=0.01,
    help="赤チャネルに含まれるわずかな近赤外光成分を減算する強度"
)

beta = st.sidebar.slider(
    "近赤外(NIR/B)ゲイン増強倍率 (β)",
    min_value=0.5, max_value=10.0, value=3.0, step=0.1,
    help="青(B)チャネルに割り当てられた近赤外感度を補正・増強する倍率"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 グラフ表示オプション")
show_normal = st.sidebar.checkbox("ノーマルカメラ（IRカット）を表示", value=False)
show_modified = st.sidebar.checkbox("物理改造後（青カット装着）を表示", value=True)
show_software = st.sidebar.checkbox("ソフト補正後を表示", value=True)


# ==========================================
# 分光感度モデルの定義 (波長 0.4 μm 〜 1.05 μm)
# ==========================================
def generate_spectral_response(alpha, beta):
    # 波長軸 (μm)
    wavelengths = np.linspace(0.4, 1.05, 500)

    # ガウス関数による基本感度の近似
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # 1. センサー本来の感度 (IRカット除去前/後の素特性)
    vis_b = gaussian(wavelengths, 0.45, 0.04)
    vis_g = gaussian(wavelengths, 0.53, 0.05)
    vis_r = gaussian(wavelengths, 0.63, 0.05)
    
    # センサー素子の近赤外応答 (特にBlue素子はNIRに強い応答を持つ)
    nir_sensor_b = 0.85 * gaussian(wavelengths, 0.85, 0.08)
    nir_sensor_r = 0.15 * gaussian(wavelengths, 0.80, 0.06)

    # フィルタ特性
    # ノーマルカメラ用 IRカットフィルタ (0.7 μm 以上をカット)
    ir_cut_filter = 1 / (1 + np.exp((wavelengths - 0.70) / 0.02))
    
    # Yubaflex用 青カットフィルタ (0.48 μm 以下をカット)
    blue_cut_filter = 1 / (1 + np.exp(-(wavelengths - 0.48) / 0.015))

    # ---- [パターン1] ノーマルカメラ (IRカット装着) ----
    norm_r = vis_r * ir_cut_filter
    norm_g = vis_g * ir_cut_filter
    norm_b = vis_b * ir_cut_filter

    # ---- [パターン2] 物理改造後 Yubaflex (青カットフィルタ装着) ----
    # Blue素子: 青可視光カット + 近赤外を受光
    mod_b = (vis_b * (1 - blue_cut_filter)) + (nir_sensor_b * blue_cut_filter)
    mod_g = vis_g
    # Red素子: 赤可視光 + わずかな近赤外
    mod_r = vis_r + (nir_sensor_r * blue_cut_filter)

    # ---- [パターン3] ソフトウェア補正後 ----
    # 赤チャネルからのNIR減算
    soft_r = np.maximum(0, mod_r - alpha * mod_b)
    soft_g = mod_g
    # 近赤外(NIR)のゲイン補正
    soft_nir = np.clip(mod_b * beta, 0, 1.0)

    return wavelengths, {
        'normal': (norm_r, norm_g, norm_b),
        'modified': (mod_r, mod_g, mod_b),
        'software': (soft_r, soft_g, soft_nir)
    }

wavelengths, spec_data = generate_spectral_response(alpha, beta)

# ==========================================
# メイン画面 タブ構成
# ==========================================
tab1, tab2 = st.tabs(["📈 分光感度特性グラフ", "🖼️ 画像処理 & NDVIシミュレーション"])

# ------------------------------------------
# TAB 1: 分光感度グラフ
# ------------------------------------------
with tab1:
    st.subheader("カメラ分光感度特性 (Relative Response vs Wavelength)")
    st.markdown("""
    **仕様のポイント:**
    - **可視域 青 (Blue)** のフィルタを除去し **近赤外 (780〜1000 nm / 0.78〜1.0 $\mu$m)** を青チャネルで受光。
    - **赤 (Red)** チャネルに含まれるわずかな近赤外の漏れをソフトウェアで減算。
    - ソフトウェア上で近赤外成分を増強（ゲイン補正）。
    """)

    fig = go.Figure()

    # ノーマルカメラ
    if show_normal:
        r, g, b = spec_data['normal']
        fig.add_trace(go.Scatter(x=wavelengths, y=r, mode='lines', name='Normal Red', line=dict(color='lightcoral', dash='dash')))
        fig.add_trace(go.Scatter(x=wavelengths, y=g, mode='lines', name='Normal Green', line=dict(color='lightgreen', dash='dash')))
        fig.add_trace(go.Scatter(x=wavelengths, y=b, mode='lines', name='Normal Blue', line=dict(color='lightblue', dash='dash')))

    # 物理改造後
    if show_modified:
        r, g, b = spec_data['modified']
        fig.add_trace(go.Scatter(x=wavelengths, y=r, mode='lines', name='改造後 Red (可視赤+微NIR)', line=dict(color='darkred', width=1.5)))
        fig.add_trace(go.Scatter(x=wavelengths, y=g, mode='lines', name='改造後 Green (可視緑)', line=dict(color='darkgreen', width=1.5)))
        fig.add_trace(go.Scatter(x=wavelengths, y=b, mode='lines', name='改造後 Blue (受光NIR)', line=dict(color='purple', width=1.5)))

    # ソフト補正後
    if show_software:
        r, g, nir = spec_data['software']
        fig.add_trace(go.Scatter(x=wavelengths, y=r, mode='lines', name='補正後 Red (純粋可視赤)', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=wavelengths, y=g, mode='lines', name='補正後 Green (純粋可視緑)', line=dict(color='green', width=3)))
        fig.add_trace(go.Scatter(x=wavelengths, y=nir, mode='lines', name='補正後 NIR (近赤外増強)', line=dict(color='magenta', width=3)))

    fig.update_layout(
        xaxis_title="Wavelength [μm]",
        yaxis_title="Relative Response",
        xaxis=dict(range=[0.4, 1.05], dtick=0.1),
        yaxis=dict(range=[0, 1.1]),
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 可視光領域と近赤外領域の背景色表示
    fig.add_vrect(x0=0.4, x1=0.7, fillcolor="gray", opacity=0.1, layer="below", line_width=0, annotation_text="可視域 (VIS)")
    fig.add_vrect(x0=0.78, x1=1.0, fillcolor="crimson", opacity=0.1, layer="below", line_width=0, annotation_text="近赤外 (NIR)")

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: 画像処理 & NDVI計算
# ------------------------------------------
with tab2:
    st.subheader("撮影画像のソフトウェア処理 & NDVI表示")
    st.write("Yubaflex撮影画像（R:赤+NIR, G:緑, B:NIR）から純粋なRed成分を抽出し、NDVIとフォールスカラーを生成します。")

    uploaded_file = st.file_uploader("Yubaflexで撮影した画像をアップロード (未選択の場合はデモ画像を生成)", type=["jpg", "jpeg", "png"])

    # サンプル画像の生成関数（植物と土壌のモデル）
    def create_synthetic_yubaflex_image():
        h, w = 300, 400
        # 画面左側に植物（緑高・NIR高）、右側に土壌（Red高・NIR低）のグラデーションを作成
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # 植物領域マスク
        veg_mask = np.clip(1.0 - xx + 0.2 * np.sin(yy * 10), 0, 1)

        # 生データ構成
        # G (緑): 可視緑
        g_raw = veg_mask * 0.6 + (1 - veg_mask) * 0.3
        # NIR: 近赤外 (植物で高い)
        nir_true = veg_mask * 0.8 + (1 - veg_mask) * 0.15
        # Red: 赤 (植物で吸収され低い, 土壌で高い)
        red_true = veg_mask * 0.15 + (1 - veg_mask) * 0.5

        # Yubaflexのセンサ応答を模倣
        # Blueチャネル = NIR
        b_channel = nir_true
        # Redチャネル = 赤 + わずかなNIR漏れ
        r_channel = red_true + 0.15 * nir_true
        # Greenチャネル = 緑
        g_channel = g_raw

        img_raw = np.stack([r_channel, g_channel, b_channel], axis=-1)
        return np.clip(img_raw * 255, 0, 255).astype(np.uint8)

    if uploaded_file is not None:
        raw_img = np.array(Image.open(uploaded_file).convert("RGB"))
    else:
        st.info("💡 テスト用のデモ画像を表示しています。")
        raw_img = create_synthetic_yubaflex_image()

    # 画像データ分解 (0.0 〜 1.0 に正規化)
    img_float = raw_img.astype(np.float32) / 255.0
    r_raw = img_float[:, :, 0]
    g_raw = img_float[:, :, 1]
    b_raw = img_float[:, :, 2] # NIR成分

    # ---- ソフトウェア補正処理 ----
    # 1. 赤チャネルからNIR漏れを減算
    r_corr = np.maximum(0, r_raw - alpha * b_raw)
    # 2. NIR成分のゲイン補正
    nir_corr = np.clip(b_raw * beta, 0, 1.0)

    # 3. NDVIの計算: (NIR - Red) / (NIR + Red)
    denominator = nir_corr + r_corr
    denominator[denominator == 0] = 1e-5 # ゼロ除算防止
    ndvi = (nir_corr -
