# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# —————— 간단 DB ——————
FOOD_DB = {
    "world_cone": {
        "name_kr": "월드콘",
        "geometry": {"circularity": 0.50, "aspect_ratio": 0.45},
        "nutrition": {"열량":"255 kcal","나트륨":"85 mg","탄수화물":"30 g","당류":"20 g","지방":"13 g","포화지방":"9 g","단백질":"4 g"},
        "allergies": ["대두","밀","우유","땅콩"]
    },
    "bbungtwigi": {
        "name_kr": "뻥튀기",
        "geometry": {"circularity": 0.90, "aspect_ratio": 1.00},
        "nutrition": {"열량":"383 kcal","나트륨":"5 mg","탄수화물":"87 g","당류":"0 g","지방":"0.5 g","포화지방":"0.1 g","단백질":"7 g"},
        "allergies": []
    },
    "demi_soda": {
        "name_kr": "데미소다 애플",
        "geometry": {"circularity": 0.65, "aspect_ratio": 0.40},
        "nutrition": {"열량":"125 kcal","나트륨":"25 mg","탄수화물":"31 g","당류":"31 g"},
        "allergies": ["사과농축과즙 함유"]
    },
    "jolly_pong": {
        "name_kr": "죠리퐁",
        "geometry": {"circularity": 0.80, "aspect_ratio": 0.75},
        "nutrition": {"열량":"325 kcal","나트륨":"100 mg","탄수화물":"60 g","당류":"29 g","지방":"6 g","포화지방":"2.7 g","단백질":"7 g"},
        "allergies": ["밀","우유","대두"]
    }
}


def detect_main_contour(
    rgb_image: np.ndarray,
    debug: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Adaptive Threshold + Closing → 외곽선 검출
    Returns:
      main_contour, debug_img (RGB with contour drawn)
    """
    try:
        gray    = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        thresh  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        main = max(contours, key=cv2.contourArea)
        debug_img = None
        if debug:
            debug_img = rgb_image.copy()
            cv2.drawContours(debug_img, [main], -1, (0,255,0), 2)
        return main, debug_img

    except Exception:
        return None, None


def extract_features(
    img: np.ndarray,
    debug: bool = False
) -> tuple[float | None, float | None, np.ndarray | None]:
    """
    contour → circularity, aspect_ratio 계산
    """
    contour, debug_img = detect_main_contour(img, debug)
    if contour is None:
        return None, None, None

    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    x,y,w,h = cv2.boundingRect(contour)
    if peri == 0 or h == 0:
        return None, None, None

    circ = (4 * np.pi * area) / (peri**2)
    ar   = w / h
    return circ, ar, debug_img


def identify_food(circ: float, ar: float) -> str | None:
    best, min_err = None, float("inf")
    for k, v in FOOD_DB.items():
        db_c, db_ar = v["geometry"]["circularity"], v["geometry"]["aspect_ratio"]
        err = abs(circ - db_c) + abs(ar - db_ar)
        if err < min_err:
            min_err, best = err, k
    return best


# —————— Streamlit UI ——————
st.title("📸 푸드 스캐너")
st.info("사진을 최대한 정면 방향으로 찍어주세요.")

uploaded = st.file_uploader("사진 업로드", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img = np.array(pil_img)

circ, ar, dbg = extract_features(img, debug=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("원본 이미지")
    st.image(pil_img, use_container_width=True)
    if dbg is not None:
        st.subheader("검출된 외곽선")
        st.image(Image.fromarray(dbg), use_container_width=True)

with col2:
    st.subheader("분석 결과")
    if circ is None:
        st.error("객체를 인식하지 못했습니다.")
    else:
        st.write(f"- Circularity: **{circ:.3f}**")
        st.write(f"- Aspect Ratio: **{ar:.3f}**")
        key = identify_food(circ, ar)
        if key:
            info = FOOD_DB[key]
            st.success(f"예측: **{info['name_kr']}**")
            with st.expander("영양 정보"):
                for n, v in info["nutrition"].items():
                    st.write(f"{n}: {v}")
            with st.expander("알레르기 정보"):
                if info["allergies"]:
                    st.warning(", ".join(info["allergies"]))
                else:
                    st.info("등록된 알레르기 정보가 없습니다.")
        else:
            st.error("데이터베이스 매칭 실패")
