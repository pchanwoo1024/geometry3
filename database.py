FOOD_DB = {
    "world_cone": {
        "name_kr": "월드콘",
        "geometry": {"circularity": 0.5, "aspect_ratio": 0.45},
        "nutrition": {
            "열량": "255 kcal", "나트륨": "85 mg", "탄수화물": "30 g", "당류": "20 g",
            "지방": "13 g", "포화지방": "9 g", "단백질": "4 g"
        },
        "allergies": ["대두", "밀", "우유", "땅콩"]
    },

-   "lotte_sand": {
-       "name_kr": "롯데샌드 오리지널",
-       "geometry": {"circularity": 0.95, "aspect_ratio": 1.0},
-       "nutrition": {
-           "열량": "265 kcal", "나트륨": "200 mg", "탄수화물": "37 g", "당류": "18 g",
-           "지방": "12 g", "포화지방": "7 g", "단백질": "3 g"
-       },
-       "allergies": ["밀", "대두", "우유"]
-   },

+   "bbungtwigi": {
+       "name_kr": "뻥튀기",
+       "geometry": {"circularity": 0.90, "aspect_ratio": 1.00},
+       "nutrition": {
+           # TODO: 실제 뻥튀기 영양 정보로 업데이트해주세요
+       },
+       "allergies": []
+   },

    "demi_soda": {
        "name_kr": "데미소다 애플",
        "geometry": {"circularity": 0.7, "aspect_ratio": 0.55},
        "nutrition": {
            "열량": "125 kcal", "나트륨": "25 mg", "탄수화물": "31 g", "당류": "31 g"
        },
        "allergies": ["사과농축과즙 함유 (별도 알레르기 유발 물질 적음)"]
    },
    "jolly_pong": {
        "name_kr": "죠리퐁",
        "geometry": {"circularity": 0.8, "aspect_ratio": 0.75},
        "nutrition": {
            "열량": "325 kcal", "나트륨": "100 mg", "탄수화물": "60 g", "당류": "29 g",
            "지방": "6 g", "포화지방": "2.7 g", "단백질": "7 g"
        },
        "allergies": ["밀", "우유", "대두"]
    }
}
