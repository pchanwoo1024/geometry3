# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# —————— 데이터베이스 직접 포함 ——————
FOOD_DB = {
    "world_cone": {
        "name_kr": "월드콘",
        "geometry": {"circularity": 0.50, "aspect_ratio": 0.45},
        "nutrition": {
            "열량": "255 kcal", "나트륨": "85 mg", "탄수화물": "30 g", "당류": "20 g",
            "지방": "13 g", "포화지방": "9 g", "단백질": "4 g"
        },
        "allergies": ["대두", "밀", "우유", "땅콩"]
    },
    "bbungtwigi": {
        "name_kr": "뻥튀기",
        "geometry": {"circularity": 0.90, "aspect_ratio": 1.00},
        "nutrition": {
            "열량": "383 kcal", "나트륨": "5 mg", "탄수화물": "87 g", "당류": "0 g",
            "지방": "0.5 g", "포화지방": "0.1 g", "단백질": "7 g"
        },
        "allergies": []
    },
    "demi_soda": {
        "name_kr": "데미소다 애플",
        "geometry": {"circularity": 0.70, "aspect_ratio": 0.55},
        "nutrition": {
            "열량": "125 kcal", "나트륨": "25 mg", "탄수화물": "31 g", "당류": "31 g"
        },
        "allergies": ["사과농축과즙 함유 (별도 알레르기 유발 물질 적음)"]
    },
    "jolly_pong": {
        "name_kr": "죠리퐁",
        "geometry": {"circularity": 0.80, "aspect_ratio": 0.75},
        "nutrition": {
            "열량": "325 kcal", "나트륨": "100 mg", "탄수화물": "60 g", "당류": "29 g",
            "지방": "6 g", "포화지방": "2.7 g", "단백질": "7 g"
        },
        "allergies": ["밀", "우유", "대두"]
    }
}


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rectify_perspective(img_array, debug=False):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_array, M, (maxW, maxH))
            if debug:
                debug_img = img_array.copy()
                cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
            return warped, debug_img

    # 못 찾으면 원본 반환
    return img_array, debug_img


def calculate_geometric_features(
    img_array, thresh_val=60, method="Fixed", debug=False
):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # 1. Threshold 선택
    if method == "Fixed":
        _, thresh = cv2.threshold(closed, thresh_val, 255, cv2.THRESH_BINARY_INV)
    elif method == "Otsu":
        _, thresh = cv2.threshold(
            closed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
    else:  # Adaptive
        thresh = cv2.adaptiveThreshold(
            closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    peri = cv2.arcLength(main_contour, True)
    x, y, w, h = cv2.boundingRect(main_contour)
    if peri == 0 or h == 0:
        return None, None, None

    circularity = (4 * np.pi * area) / (peri ** 2)
    aspect_ratio = w / h

    debug_img = None
    if debug:
        debug_img = img_array.copy()
        cv2.drawContours(debug_img, [main_contour], -1, (0, 255, 0), 2)

    return circularity, aspect_ratio, debug_img


def identify_food(circularity, aspect_ratio):
    min_err, best = float("inf"), None
    for k, d in FOOD_DB.items():
        err = abs(circularity - d["geometry"]["circularity"]) + abs(aspect_ratio - d["geometry"]["aspect_ratio"])
        if err < min_err:
            min_err, best = err, k
    return best


# —————— Streamlit UI ——————
st.title("📸 식품 영양 분석기 (향상된 객체 검출)")
st.write("▶ 원근 보정 + 여러 임계치 방식 지원 + 디버그 출력")
st.sidebar.header("Threshold 세팅")
method = st.sidebar.selectbox("방식 선택", ["Fixed", "Otsu", "Adaptive"])
th = st.sidebar.slider("Fixed Threshold 값", 0, 255, 60) if method == "Fixed" else None

uploaded = st.file_uploader("사진 업로드...", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
orig = np.array(pil_img)

# 1) Perspective 보정
warped, dbg_rect = rectify_perspective(orig, debug=True)

# 2) 특징 계산
circ, ar, dbg_cnt = calculate_geometric_features(warped, th or 0, method, debug=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("원본")
    st.image(pil_img, use_container_width=True)
    if dbg_rect is not None:
        st.subheader("검출된 사각형")
        st.image(Image.fromarray(dbg_rect), use_container_width=True)
    st.subheader("보정된 이미지")
    st.image(Image.fromarray(warped), use_container_width=True)
    if dbg_cnt is not None:
        st.subheader("검출된 윤곽선")
        st.image(Image.fromarray(dbg_cnt), use_container_width=True)

with col2:
    st.subheader("분석")
    if circ is None:
        st.error("객체를 인식하지 못했습니다. Threshold 방식을 바꿔 보세요.")
    else:
        st.write(f"- Circularity: **{circ:.3f}**")
        st.write(f"- Aspect Ratio: **{ar:.3f}**")
        key = identify_food(circ, ar)
        if key:
            info = FOOD_DB[key]
            st.success(f"이 과자는 **{info['name_kr']}** 로 추정됩니다!")
            with st.expander("영양 정보"):
                for n, v in info["nutrition"].items():
                    st.write(f"{n}: {v}")
            with st.expander("알레르기 정보"):
                if info["allergies"]:
                    st.warning(", ".join(info["allergies"]))
                else:
                    st.info("등록된 알레르기 정보가 없습니다.")
        else:
            st.error("데이터베이스에서 일치하는 과자를 찾지 못했습니다.")
