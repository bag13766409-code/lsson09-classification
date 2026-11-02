
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# ==========================================
# 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(error)
    st.stop()


# ==========================================
# 데이터 전처리
# ==========================================
# 분석에 필요한 열을 숫자형으로 변환
numeric_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "stroke",
    "bmi"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# 뇌졸중 여부 이름 붙이기
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# ==========================================
# 페이지 제목
# ==========================================
st.title("🔎 데이터 탐색")

st.write("""
여러 그래프와 표를 통해 뇌졸중 데이터의 특징을 살펴봅니다.

나이, 혈당, 고혈압, 심장병, BMI, 흡연 상태에 따라
데이터가 어떻게 달라지는지 확인해 보세요.
""")


# ==========================================
# 1. 나이와 평균 혈당의 분포
# ==========================================
st.header("1. 나이와 평균 혈당의 분포")

st.write("""
히스토그램은 데이터가 어떤 구간에 많이 모여 있는지
확인할 수 있는 그래프입니다.
""")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이의 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
        color_discrete_sequence=["#4C78A8"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당의 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#F58518"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


# ==========================================
# 2. 뇌졸중 여부에 따른 나이와 혈당 비교
# ==========================================
st.header("2. 뇌졸중 여부에 따른 비교")

st.write("""
상자그림을 이용하면 뇌졸중을 겪은 사람과
겪지 않은 사람의 나이와 평균 혈당 분포를 비교할 수 있습니다.

상자 안의 선은 중앙값을 나타냅니다.
""")

# 뇌졸중 여부가 확인되는 데이터만 사용
stroke_df = df.dropna(
    subset=["stroke_group"]
).copy()

stroke_order = [
    "뇌졸중 없음",
    "뇌졸중 있음"
]

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        stroke_df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        category_orders={
            "stroke_group": stroke_order
        },
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이"
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이"
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )

with col2:
    fig_glucose_box = px.box(
        stroke_df,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        category_orders={
            "stroke_group": stroke_order
        },
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# ==========================================
# 두 그룹의 평균값 표
# ==========================================
st.subheader("뇌졸중 여부에 따른 평균값")

mean_table = (
    stroke_df
    .groupby("stroke_group", observed=False)
    .agg(
        나이_평균=("age", "mean"),
        평균혈당_평균=("avg_glucose_level", "mean"),
        사람_수=("stroke", "count")
    )
    .reindex(stroke_order)
    .reset_index()
)

mean_table.columns = [
    "뇌졸중 여부",
    "평균 나이",
    "평균 혈당",
    "사람 수"
]

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 3. 고혈압 여부와 뇌졸중 비율
# ==========================================
st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")

st.write("""
고혈압과 심장병이 있는 그룹과 없는 그룹을 나누어
각 그룹에서 뇌졸중을 겪은 사람의 비율을 비교합니다.
""")

def make_stroke_rate_table(data, column):
    temp = data.dropna(
        subset=[column, "stroke"]
    ).copy()

    temp["그룹"] = temp[column].map({
        0: "없음",
        1: "있음"
    })

    temp = temp.dropna(subset=["그룹"])

    result = (
        temp.groupby("그룹")
        .agg(
            전체_사람=("stroke", "count"),
            뇌졸중_사람=("stroke", "sum"),
            뇌졸중_비율=("stroke", "mean")
        )
        .reindex(["없음", "있음"])
        .reset_index()
    )

    result["뇌졸중 비율(%)"] = (
        result["뇌졸중_비율"] * 100
    ).round(2)

    return result


# 고혈압
hypertension_table = make_stroke_rate_table(
    df,
    "hypertension"
)

col1, col2 = st.columns(2)

with col1:
    fig_hypertension = px.bar(
        hypertension_table,
        x="그룹",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={
            "그룹": "고혈압",
            "뇌졸중 비율(%)": "뇌졸중 비율 (%)"
        },
        color="그룹",
        color_discrete_map={
            "없음": "#4C78A8",
            "있음": "#E45756"
        }
    )

    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_hypertension.update_layout(
        showlegend=False,
        yaxis_title="뇌졸중 비율 (%)",
        xaxis_title="고혈압 여부"
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )


# 심장병
heart_table = make_stroke_rate_table(
    df,
    "heart_disease"
)

with col2:
    fig_heart = px.bar(
        heart_table,
        x="그룹",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={
            "그룹": "심장병",
            "뇌졸중 비율(%)": "뇌졸중 비율 (%)"
        },
        color="그룹",
        color_discrete_map={
            "없음": "#4C78A8",
            "있음": "#E45756"
        }
    )

    fig_heart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_heart.update_layout(
        showlegend=False,
        yaxis_title="뇌졸중 비율 (%)",
        xaxis_title="심장병 여부"
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )


# ==========================================
# 4. BMI 빈 값 분석
# ==========================================
st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")

st.write("""
BMI 값이 비어 있는 사람들만 따로 살펴봅니다.

이 사람들의 뇌졸중 비율과 전체 데이터의 뇌졸중 비율을
비교해 보세요.
""")

# BMI가 비어 있는 사람
missing_bmi_df = df[df["bmi"].isna()]

# 전체 사람
total_stroke_df = df.dropna(
    subset=["stroke"]
)

# BMI 빈 값인 사람 중 stroke 값이 있는 사람
missing_bmi_stroke_df = missing_bmi_df.dropna(
    subset=["stroke"]
)

missing_bmi_people = len(missing_bmi_df)

missing_bmi_stroke_count = int(
    (missing_bmi_stroke_df["stroke"] == 1).sum()
)

missing_bmi_rate = (
    missing_bmi_stroke_df["stroke"].mean() * 100
    if len(missing_bmi_stroke_df) > 0
    else 0
)

overall_stroke_count = int(
    (total_stroke_df["stroke"] == 1).sum()
)

overall_stroke_rate = (
    total_stroke_df["stroke"].mean() * 100
    if len(total_stroke_df) > 0
    else 0
)

bmi_comparison = pd.DataFrame({
    "비교 대상": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        missing_bmi_people,
        len(total_stroke_df)
    ],
    "뇌졸중을 겪은 사람 수": [
        missing_bmi_stroke_count,
        overall_stroke_count
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_rate, 2),
        round(overall_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 5. 흡연 상태별 사람 수
# ==========================================
st.header("5. 흡연 상태별 사람 수")

st.write("""
흡연 상태의 값마다 몇 명의 사람이 있는지 확인합니다.

빈 값도 별도로 표시합니다.
""")

smoking_table = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_table = smoking_table.sort_values(
    "사람 수",
    ascending=False
).reset_index(drop=True)

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 페이지 하단 안내
# ==========================================
st.divider()

st.caption("""
※ 이 페이지는 데이터의 분포와 그룹별 차이를 탐색하기 위한 교육용 자료입니다.
그래프에서 발견한 차이만으로 원인과 결과를 단정할 수는 없습니다.
""")
