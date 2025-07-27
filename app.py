# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from database import FOOD_DB # database.py에서 식품 정보 가져오기

def calculate_geometric_features(image):
    """
    이미지에서 객체의 윤곽선을 찾아 원형도와 가로세로 비율을 계산합니다.
    """
    # 이미지를 OpenCV에서 처리할 수 있는 형식으로 변환
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 이미지 블러 처리 및 이진화
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # 가장 큰 윤곽선을 객체로 간주
    main_contour = max(contours, key=cv2.contourArea)

    # 특징 계산
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    x, y, w, h = cv2.boundingRect(main_contour)

    if perimeter == 0 or h == 0:
        return None, None

    # 원형도 계산
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    # 가로세로 비율 계산
    aspect_ratio = w / h

    return circularity, aspect_ratio

def identify_food(circularity, aspect_ratio):
    """
    계산된 특징과 DB를 비교하여 가장 유사한 식품을 찾습니다.
    """
    min_error = float('inf')
    identified_food_key = None

    for key, value in FOOD_DB.items():
        db_circularity = value["geometry"]["circularity"]
        db_aspect_ratio = value["geometry"]["aspect_ratio"]
        
        # 오차 계산 (단순 차이의 합)
        error = abs(circularity - db_circularity) + abs(aspect_ratio - db_aspect_ratio)

        if error < min_error:
            min_error = error
            identified_food_key = key
            
    return identified_food_key


# --- Streamlit UI 구성 ---

st.title("📸 식품 영양 정보 분석기")
st.write("식품 사진을 업로드하면 어떤 식품인지 분석하고 영양 정보를 알려드립니다.")
st.info("정면에서 찍은, 배경이 단순한 사진일수록 인식률이 높습니다.")

# ※ Homography 보정은 복잡성이 높아 이 코드에서는 생략되었습니다.
# 계획하신 대로 Homography 로직을 추가한다면, 아래 'uploaded_file' 처리 부분에 넣으시면 됩니다.

uploaded_file = st.file_uploader("여기에 사진을 올려주세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("업로드한 사진")
        st.image(image, use_container_width=True)

    # 기하학적 특징 계산
    circularity, aspect_ratio = calculate_geometric_features(image)
    
    with col2:
        st.subheader("분석 결과")
        if circularity is not None and aspect_ratio is not None:
            # 식품 식별
            food_key = identify_food(circularity, aspect_ratio)
            
            if food_key:
                result = FOOD_DB[food_key]
                st.success(f"이 식품은 **{result['name_kr']}** 같아요!")
                
                # 영양 정보 표시
                with st.expander("영양 정보 보기"):
                    for nutrient, value in result["nutrition"].items():
                        st.text(f"{nutrient}: {value}")

                # 알레르기 정보 표시
                with st.expander("알레르기 정보 보기"):
                    st.warning(", ".join(result["allergies"]))
            else:
                st.error("데이터베이스에서 일치하는 식품을 찾지 못했습니다.")
        else:
            st.error("이미지에서 객체를 인식하지 못했습니다. 더 선명한 사진을 사용해보세요.")
