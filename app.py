from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import streamlit as st

st.set_page_config(layout="wide")


def calculate_ndvi(image_array):
    b, g, r = cv2.split(image_array)
    r_float = r.astype(float)
    b_float = b.astype(float)

    denominator = r_float + b_float
    ndvi = np.divide(
        r_float - b_float,
        denominator,
        out=np.zeros_like(r_float),
        where=denominator != 0,
    )
    return ndvi


def create_ndvi_colormap(ndvi):
    ndvi_normalized = ((ndvi + 1.0) / 2.0 * 255).astype(np.uint8)
    ndvi_color = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    return cv2.cvtColor(ndvi_color, cv2.COLOR_BGR2RGB)


def extract_vegetation(img_array, ndvi, threshold):
    mask = ndvi >= threshold
    extracted_img = img_array.copy()
    extracted_img[~mask] = [0, 0, 0]

    total_pixels = mask.size
    vegetation_pixels = np.sum(mask)
    area_ratio = (vegetation_pixels / total_pixels) * 100

    return extracted_img, mask, area_ratio


@st.cache_data
def load_image_from_url(url):
    # GitHubのblob URLをraw URLに自動変換して画像を取得
    raw_url = url.replace(
        "github.com", "raw.githubusercontent.com"
    ).replace("/blob/", "/")
    response = requests.get(raw_url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


st.title("NDVI 解析 & 相対応答グラフビューア")

# サイドバー設定
with st.sidebar:
    st.header("1. 画像選択")

    input_source = st.radio(
        "画像の入力方法を選択してください",
        ("サンプル画像を使用", "手元から画像をアップロード"),
    )

    image = None

    if input_source == "サンプル画像を使用":
        sample_options = {
            "NDVItest.jpg (GitHub)": "https://github.com/yasaibusisetuengei/NDVI_analysis/blob/main/NDVItest.jpg"
        }
        selected_sample = st.selectbox(
            "サンプル画像を選択", list(sample_options.keys())
        )

        try:
            sample_url = sample_options[selected_sample]
            image = load_image_from_url(sample_url)
            st.success("サンプル画像を読み込みました")
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {e}")

    else:
        uploaded_file = st.file_uploader(
            "対象の画像をアップロードしてください", type=["jpg", "png", "jpeg"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    st.subheader("2. 植生抽出パラメータ")
    ndvi_threshold = st.slider(
        "NDVI 閾値 (Threshold)",
        min_value=-1.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="健康な葉領域を分離する閾値です（目安: 0.2〜0.5以上）。",
    )

if image is not None:
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # NDVI計算・処理
    ndvi_data = calculate_ndvi(img_bgr)
    ndvi_colored = create_ndvi_colormap(ndvi_data)
    extracted_img, mask, area_ratio = extract_vegetation(
        img_array, ndvi_data, ndvi_threshold
    )

    st.subheader("1. NDVI 画像解析結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="オリジナル画像", use_column_width=True)
    with col2:
        st.image(
            ndvi_colored,
            caption="NDVI 疑似カラー画像 (JET)",
            use_column_width=True,
        )
    with col3:
        st.image(
            extracted_img,
            caption=f"抽出領域 (閾値 >= {ndvi_threshold:.2f})",
            use_column_width=True,
        )
        st.metric(
            label="抽出領域の面積比率", value=f"{area_ratio:.1f} %"
        )

    st.subheader("2. 相対応答 (Relative Response) グラフ")

    # グラフデータ作成
    wavelength = np.linspace(0.4, 1.0, 100)
    response_vis = np.exp(-(((wavelength - 0.45) / 0.05) ** 2))
    response_nir = np.exp(-(((wavelength - 0.82) / 0.08) ** 2))
    relative_response = np.maximum(response_vis, response_nir) * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(
        wavelength,
        relative_response,
        color="purple",
        linewidth=2,
        label="Dual-band Filter Response",
    )

    # 帯域の背景色づけ（視覚的理解を助ける）
    ax.axvspan(0.40, 0.50, color="blue", alpha=0.1, label="Blue Visible Peak")
    ax.axvspan(0.75, 0.90, color="red", alpha=0.1, label="NIR Peak")

    ax.set_xlabel("wavelength [micrometer]")
    ax.set_ylabel("relative response [%]")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    st.pyplot(fig)

    # 相対応答グラフの詳しい解説アコーディオン
    with st.expander("💡 相対応答 (Relative Response) グラフとは？（詳細解説）"):
        st.markdown("""
        ### 波長と相対応答の仕組み
        このグラフは、カメラに装着された**デュアルバンドパスフィルター**が「どの波長の光をどの程度透過させるか（感度特性）」を示したものです。

        * **横軸：波長 (`wavelength [micrometer]`)**
          * **0.40 〜 0.50 μm (可視光・青色領域)**：植物に含まれる色素（クロロフィル）が強く吸収する光の波長域です。
          * **0.75 〜 0.90 μm (近赤外線・NIR領域)**：植物の葉の海綿状組織が強い反射特性を持つ波長域です。人間の目には見えません。
        
        * **縦軸：相対応答 (`relative response [%]`)**
          * センサーが特定の波長光を受け取る割合（感度ピーク）を表します。

        ---

        ### なぜ1枚の画像でNDVIが計算できるのか？
        一般的なカメラの赤外線（IR）カットフィルターを取り外し、このような**特定の2つの波長だけを通すフィルター**を取り付けることで、以下の割り当てが可能になります：
        
        1. **R (赤) チャンネル** ➔ フィルター透過後の**近赤外線 (NIR)** 光を受光
        2. **B (青) チャンネル** ➔ フィルター透過後の**可視光 (Blue)** 光を受光

        これにより、特殊なマルチスペクトルカメラを使わずとも、通常カメラの1回の撮影で得られる画像（R値とB値）から **`NDVI = (R - B) / (R + B)`** の演算が可能となります。
        """)
else:
    st.info(
        "サイドバーから「サンプル画像を使用」を選択するか、画像をアップロードしてください。"
    )
