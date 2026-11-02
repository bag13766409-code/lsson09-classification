
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 기본 설정
# ==========================================
APP_TITLE = "뇌졸중 예측 실습실"

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_예측_실습실.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# 기록 불러오기
# ==========================================
def load_records():
    if not RECORD_FILE.exists():
        return {
            "meanings": {},
            "source": ""
        }

    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "meanings": data.get("meanings", {}),
            "source": data.get("source", "")
        }

    except (json.JSONDecodeError, OSError):
        return {
            "meanings": {},
            "source": ""
        }


# ==========================================
# 기록 저장하기
# ==========================================
def save_records(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


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
# 필요한 열 확인
# ==========================================
required_columns = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


records = load_records()


# ==========================================
# 앱 제목
# ==========================================
st.title("🧠 뇌졸중 예측 실습실")

st.caption(
    "데이터를 탐색하고 뇌졸중 예측을 준비하는 실습 공간입니다."
)


# ==========================================
# 1. 데이터 소개
# ==========================================
st.header("1. 데이터 소개")

st.write("""
이 데이터는 사람들의 인구학적 정보와 건강 관련 정보를 담고 있습니다.

각 사람의 나이, 성별, 혈압, 심장 질환, 혈당, BMI,
흡연 상태 등의 정보를 살펴볼 수 있습니다.

마지막의 stroke 열은 뇌졸중 발생 여부를 나타냅니다.

이 데이터를 이용해 뇌졸중과 관련된 특징을 탐색하고
예측 모델을 만들어 볼 수 있습니다.
""")


# ==========================================
# 2. 숫자 카드
# ==========================================
total_people = len(df)

column_count = len(df.columns)

stroke_count = int(
    (pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()
)

stroke_rate = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 전체 사람 수",
    f"{total_people:,}명"
)

col2.metric(
    "📋 열 개수",
    f"{column_count:,}개"
)

col3.metric(
    "🧠 stroke가 1인 사람 수",
    f"{stroke_count:,}명"
)

col4.metric(
    "📊 stroke가 1인 비율",
    f"{stroke_rate:.2f}%"
)

st.caption(
    "※ stroke가 1인 사람 수와 비율은 데이터의 stroke 열을 기준으로 계산합니다."
)


# ==========================================
# 3. 열 정보 표
# ==========================================
st.header("2. 열 이름과 변수 정보")

st.write("""
각 열이 어떤 의미인지 교재를 참고하여
우리말 뜻을 직접 입력해 보세요.
""")

column_info = []

for column in df.columns:

    if pd.api.types.is_numeric_dtype(df[column]):
        value_kind = "수치형"
    else:
        value_kind = "범주형"

    column_info.append({
        "열 이름": column,
        "우리말 뜻": records["meanings"].get(column, ""),
        "값의 종류": value_kind,
        "빈 값 개수": int(df[column].isna().sum())
    })


info_df = pd.DataFrame(column_info)

edited_info = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 직접 입력하세요.",
            width="large"
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="column_info_editor"
)


# BMI 빈 값 안내
bmi_missing = int(df["bmi"].isna().sum())

if bmi_missing > 0:
    st.info(
        f"bmi 열에는 빈 값이 {bmi_missing:,}개 있습니다."
    )


# ==========================================
# 4. 데이터 미리 보기
# ==========================================
st.header("3. 데이터 처음 다섯 줄")

st.write(
    "CSV 파일에 들어 있는 처음 다섯 줄을 그대로 보여 줍니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 5. 데이터 출처
# ==========================================
st.header("4. 데이터 출처")

st.write(
    "교재에 적힌 데이터 출처를 아래에 입력하세요."
)

source_text = st.text_area(
    "데이터 출처",
    value=records["source"],
    placeholder="교재에 나온 데이터 출처를 입력하세요.",
    height=120,
    key="source_editor"
)


# ==========================================
# 6. 기록 저장 버튼
# ==========================================
if st.button(
    "💾 우리말 뜻과 출처 저장",
    type="primary",
    use_container_width=True
):

    meanings_to_save = {
        str(row["열 이름"]): str(row["우리말 뜻"])
        for _, row in edited_info.iterrows()
    }

    try:
        save_records(
            meanings_to_save,
            source_text
        )

        st.success(
            "우리말 뜻과 데이터 출처를 저장했습니다!"
        )

    except OSError as error:
        st.error("파일 저장에 실패했습니다.")
        st.exception(error)


st.caption(
    "저장 위치: 내_기록/뇌졸중_예측_실습실.json"
)


# ==========================================
# 7. 안내
# ==========================================
with st.expander("실습 안내"):

    st.write("""
    이 앱은 데이터 분석과 머신러닝 학습을 위한 교육용 도구입니다.

    실제 의료 진단이나 치료 판단을 대신할 수 없습니다.
    """)

    st.write("데이터 출처 주소:")

    st.code(DATA_URL)
