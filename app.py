# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# —————— 데이터베이스 (FOOD_DB) 직접 포함 ——————
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


def calculate_geometric_features(image: Image.Image, debug: bool = False):
    """
    - 이미지를 OpenCV 형식으로 변환한 뒤
    - 그레이스케일 → 블러 → 단일 임계치 이진화 → 외곽선 추출
    - 가장 큰 contour로부터 원형도(circularity)와 가로세로비(aspect_ratio) 계산
    - debug=True 시, 외곽선이 그려진 디버그 이미지도 함께 리턴
    """
    # PIL → NumPy
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    x, y, w, h = cv2.boundingRect(main_contour)

    if perimeter == 0 or h == 0:
        return None, None, None

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    aspect_ratio = w / h

    debug_img = None
    if debug:
        debug_img = img_array.copy()
        cv2.drawContours(debug_img, [main_contour], -1, (0, 255, 0), 2)

    return circularity, aspect_ratio, debug_img


def identify_food(circularity: float, aspect_ratio: float):
    """
    계산된 특징과 FOOD_DB를 비교하여 오차(error)가 가장 적은 키를 반환
    """
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
st.title("📸 식품 영양 정보 분석기 (단일 파일 버전)")
st.write("— 배경과 조명 영향을 크게 받지만, 간단한 원형도/비율 비교로 4가지 과자를 분류합니다.")
st.info("정면에서 찍은, 배경이 단순한 사진일수록 인식률이 높습니다.")

uploaded_file = st.file_uploader("📂 사진 업로드...", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

# 이미지 로드
image = Image.open(uploaded_file).convert("RGB")

# 기하 특징 + debug 이미지 얻기
circularity, aspect_ratio, debug_img = calculate_geometric_features(image, debug=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("원본 이미지")
    st.image(image, use_container_width=True)

    if debug_img is not None:
        st.subheader("검출된 외곽선 (디버그)")
        st.image(Image.fromarray(debug_img), use_container_width=True)

with col2:
    st.subheader("분석 결과")
    if circularity is None or aspect_ratio is None:
        st.error("객체를 인식하지 못했습니다. 더 선명한 사진을 사용해 보세요.")
    else:
        st.write(f"- 원형도 (Circularity): **{circularity:.3f}**")
        st.write(f"- 가로세로 비 (Aspect Ratio): **{aspect_ratio:.3f}**")

        food_key = identify_food(circularity, aspect_ratio)
        if food_key:
            info = FOOD_DB[food_key]
            st.success(f"이 과자는 **{info['name_kr']}** 로 추정됩니다!")
            with st.expander("✅ 영양 정보 보기"):
                for nut, val in info["nutrition"].items():
                    st.text(f"{nut}: {val}")
            with st.expander("⚠️ 알레르기 정보 보기"):
                if info["allergies"]:
                    st.warning(", ".join(info["allergies"]))
                else:
                    st.info("등록된 알레르기 정보가 없습니다.")
        else:
            st.error("데이터베이스와 일치하는 과자를 찾지 못했습니다.")
