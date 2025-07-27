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
    """
    - 입력 NumPy 이미지에서 가장 큰 4각형 윤곽을 찾아
    - 원근 보정(homography)으로 사영 이미지를 반환합니다.
    - debug=True 시, 검출된 사각형 외곽을 그려둔 디버그 이미지도 함께 반환합니다.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
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

    # 사영 보정 실패 시 원본 반환
    return img_array, debug_img


def calculate_geometric_features(img_array, debug=False):
    """
    - 이미지에서 외곽선을 찾아
    - 원형도와 가로세로 비율을 계산합니다.
    - debug=True 시, 검사된 윤곽선을 그린 디버그 이미지를 반환합니다.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 모폴로지 클로징으로 구멍 메우기
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(closed, 60, 255, cv2.THRESH_BINARY_INV)

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
        debug_img = cv2.cvtColor(img_array.copy(), cv2.COLOR_RGB2BGR)
        cv2.drawContours(debug_img, [main_contour], -1, (0, 255, 0), 2)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

    return circularity, aspect_ratio, debug_img


def identify_food(circularity, aspect_ratio):
    min_error = float("inf")
    best_key = None
    for key, data in FOOD_DB.items():
        db_circ = data["geometry"]["circularity"]
        db_ar = data["geometry"]["aspect_ratio"]
        error = abs(circularity - db_circ) + abs(aspect_ratio - db_ar)
        if error < min_error:
            min_error = error
            best_key = key
    return best_key


# —————— Streamlit UI ——————
st.title("📸 식품 영양 분석기 (원근 보정 + 디버그)")
st.write("Homography 보정과 모폴로지 후처리로 객체 검출 정확도를 개선했습니다.")
st.info("정면에서 찍은 사진일수록, 가장 큰 사각형 윤곽을 올바르게 찾아냅니다.")

uploaded_file = st.file_uploader("📂 사진 업로드...", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

# 원본 PIL → NumPy
pil_img = Image.open(uploaded_file).convert("RGB")
orig = np.array(pil_img)

# 1) 원근 보정
warped, debug_rect = rectify_perspective(orig, debug=True)

# 2) 특징 계산 (warped 기준)
circularity, aspect_ratio, debug_contour = calculate_geometric_features(warped, debug=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("원본 이미지")
    st.image(pil_img, use_container_width=True)

    if debug_rect is not None:
        st.subheader("사영 보정 (검출된 사각형)")
        st.image(Image.fromarray(debug_rect), use_container_width=True)

    st.subheader("보정 후 이미지")
    st.image(Image.fromarray(warped), use_container_width=True)

    if debug_contour is not None:
        st.subheader("검출된 외곽선 (디버그)")
        st.image(Image.fromarray(debug_contour), use_container_width=True)

with col2:
    st.subheader("분석 결과")
    if circularity is None:
        st.error("객체를 인식하지 못했습니다.")
    else:
        st.write(f"- 원형도 (Circularity): **{circularity:.3f}**")
        st.write(f"- 가로세로 비 (Aspect Ratio): **{aspect_ratio:.3f}**")
        key = identify_food(circularity, aspect_ratio)
        if key:
            info = FOOD_DB[key]
            st.success(f"이 과자는 **{info['name_kr']}** 로 추정됩니다!")
            with st.expander("✅ 영양 정보"):
                for nut, val in info["nutrition"].items():
                    st.write(f"{nut}: {val}")
            with st.expander("⚠️ 알레르기 정보"):
                if info["allergies"]:
                    st.warning(", ".join(info["allergies"]))
                else:
                    st.info("등록된 알레르기 정보가 없습니다.")
        else:
            st.error("데이터베이스에서 일치하는 과자를 찾지 못했습니다.")
