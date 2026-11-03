import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


st.title("📋 채점표")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "BMI",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

MODEL_NAMES = {
    "logistic": "로지스틱 회귀(확률로 답하는 모델)",
    "tree": "의사결정트리(질문으로 답하는 모델)",
}


# 1. 데이터 읽기
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)

    required_columns = [
        "id", "age", "avg_glucose_level", "bmi",
        "hypertension", "heart_disease", "stroke"
    ]

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=[
            "id", "age", "avg_glucose_level",
            "hypertension", "heart_disease", "stroke"
        ]
    )

    data = data.sort_values("id").reset_index(drop=True)

    return data


# 2. 데이터 나누기
def split_data(data):
    test_mask = np.arange(len(data)) % 10 < 3

    train_data = data.loc[~test_mask].copy()
    test_data = data.loc[test_mask].copy()

    return train_data, test_data


# 3. 평가 지표 계산
def calculate_scores(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0

    actual_positive = tp + fn
    recall = tp / actual_positive if actual_positive > 0 else None

    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive > 0 else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def format_score(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.4f}"


# 4. 혼동행렬 표시
def show_confusion_matrix(scores):

    tn = scores["TN"]
    fp = scores["FP"]
    fn = scores["FN"]
    tp = scores["TP"]

    matrix = pd.DataFrame(
        {
            "예측: 뇌졸중 아님 (0)": [
                f"TN: {tn}명",
                f"FN: {fn}명",
                f"{tn + fn}명"
            ],
            "예측: 뇌졸중 (1)": [
                f"FP: {fp}명",
                f"TP: {tp}명",
                f"{fp + tp}명"
            ],
            "행 합계": [
                f"{tn + fp}명",
                f"{fn + tp}명",
                f"{tn + fp + fn + tp}명"
            ]
        },
        index=[
            "실제: 뇌졸중 아님 (0)",
            "실제: 뇌졸중 (1)",
            "열 합계"
        ]
    )

    st.table(matrix)


# 5. 환자 목록 만들기
def prepare_patient_list(df):

    columns = [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ]

    result = df[columns].copy()

    result = result.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부"
    })

    result["고혈압 여부"] = result["고혈압 여부"].map({
        1: "있음",
        0: "없음"
    })

    result["심장병 여부"] = result["심장병 여부"].map({
        1: "있음",
        0: "없음"
    })

    result["나이"] = result["나이"].round(1)
    result["평균 혈당"] = result["평균 혈당"].round(1)

    return result


# --------------------------------------------------
# 화면 시작
# --------------------------------------------------

st.write(
    "테스트 데이터로 두 분류 모델이 뇌졸중 환자를 "
    "얼마나 잘 찾아내는지 비교합니다."
)

st.divider()

# 데이터 읽기
try:
    data = load_data()
except Exception as e:
    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


# 특징 선택
st.subheader("1. 모델 학습")

selected_features = st.multiselect(
    "학습에 사용할 특징을 선택하세요.",
    options=list(FEATURE_LABELS.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_LABELS[x]
)

if not selected_features:
    st.warning("특징을 하나 이상 선택하세요.")
    st.stop()


# 데이터 분리
train_data, test_data = split_data(data)

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

test_positive_count = int((y_test == 1).sum())

st.info(
    f"테스트용 사람 수: **{len(test_data):,}명** "
    f"| 그중 실제 뇌졸중 환자: **{test_positive_count:,}명**"
)


# BMI 결측값 처리
if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("BMI 데이터에 유효한 값이 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# --------------------------------------------------
# 6. 로지스틱 회귀 학습
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=2000)

logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)


# --------------------------------------------------
# 7. 의사결정트리 학습
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)


# --------------------------------------------------
# 8. 기준 모델
# --------------------------------------------------

majority_class = int(y_train.value_counts().idxmax())

baseline_pred = np.full(
    len(y_test),
    majority_class
)


# --------------------------------------------------
# 9. 평가 결과 계산
# --------------------------------------------------

logistic_scores = calculate_scores(y_test, logistic_pred)
tree_scores = calculate_scores(y_test, tree_pred)
baseline_scores = calculate_scores(y_test, baseline_pred)


# --------------------------------------------------
# 10. 혼동행렬
# --------------------------------------------------

st.divider()
st.subheader("2. 혼동행렬")

st.write("행은 실제 정답이고, 열은 모델의 예측입니다.")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**{MODEL_NAMES['logistic']}**")
    show_confusion_matrix(logistic_scores)

with col2:
    st.markdown(f"**{MODEL_NAMES['tree']}**")
    show_confusion_matrix(tree_scores)


# --------------------------------------------------
# 11. 평가 지표 비교
# --------------------------------------------------

st.divider()
st.subheader("3. 평가 지표 비교")

st.write("""
- 정확도: 전체 사람 중 정답을 맞힌 비율
- 재현율: 실제 뇌졸중 환자 중 찾아낸 비율
- 정밀도: 뇌졸중이라고 예측한 사람 중 실제 환자 비율
- F1: 정밀도와 재현율의 조화평균
""")

metrics_table = pd.DataFrame([
    {
        "모델": MODEL_NAMES["logistic"],
        "정확도": format_score(logistic_scores["정확도"]),
        "재현율": format_score(logistic_scores["재현율"]),
        "정밀도": format_score(logistic_scores["정밀도"]),
        "F1": format_score(logistic_scores["F1"])
    },
    {
        "모델": MODEL_NAMES["tree"],
        "정확도": format_score(tree_scores["정확도"]),
        "재현율": format_score(tree_scores["재현율"]),
        "정밀도": format_score(tree_scores["정밀도"]),
        "F1": format_score(tree_scores["F1"])
    },
    {
        "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도": format_score(baseline_scores["정확도"]),
        "재현율": format_score(baseline_scores["재현율"]),
        "정밀도": format_score(baseline_scores["정밀도"]),
        "F1": format_score(baseline_scores["F1"])
    }
])

st.dataframe(
    metrics_table,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "뇌졸중이라고 예측한 사람이 0명이면 "
    "정밀도와 F1은 계산할 수 없습니다."
)


# --------------------------------------------------
# 12. 놓친 환자와 헛짚은 사람
# --------------------------------------------------

st.divider()
st.subheader("4. 놓친 환자와 헛짚은 사람")

st.write("""
- 놓친 환자(FN): 실제 뇌졸중 환자인데 아니라고 예측한 사람
- 헛짚은 사람(FP): 실제 환자가 아닌데 뇌졸중이라고 예측한 사람
""")

selected_model = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=["logistic", "tree"],
    format_func=lambda x: MODEL_NAMES[x]
)

if selected_model == "logistic":
    selected_pred = logistic_pred
    selected_scores = logistic_scores
else:
    selected_pred = tree_pred
    selected_scores = tree_scores


result_data = test_data.copy()

result_data["실제"] = y_test.to_numpy()
result_data["예측"] = selected_pred


# 놓친 환자
missed_patients = result_data[
    (result_data["실제"] == 1) &
    (result_data["예측"] == 0)
].sort_values("age", ascending=False)

# 헛짚은 사람
false_alarm_patients = result_data[
    (result_data["실제"] == 0) &
    (result_data["예측"] == 1)
].sort_values("age", ascending=False)


missed_count = len(missed_patients)
false_alarm_count = len(false_alarm_patients)


col1, col2 = st.columns(2)

with col1:

    st.markdown(
        f"**놓친 환자 (FN) — 전체 {missed_count:,}명**"
    )

    if missed_count == 0:
        st.success("놓친 환자가 없습니다.")
    else:
        st.dataframe(
            prepare_patient_list(missed_patients.head(20)),
            hide_index=True,
            use_container_width=True
        )

        if missed_count > 20:
            st.caption(
                f"전체 {missed_count:,}명 중 "
                "나이가 많은 순서로 20명만 표시했습니다."
            )


with col2:

    st.markdown(
        f"**헛짚은 사람 (FP) — 전체 {false_alarm_count:,}명**"
    )

    if false_alarm_count == 0:
        st.success("헛짚은 사람이 없습니다.")
    else:
        st.dataframe(
            prepare_patient_list(false_alarm_patients.head(20)),
            hide_index=True,
            use_container_width=True
        )

        if false_alarm_count > 20:
            st.caption(
                f"전체 {false_alarm_count:,}명 중 "
                "나이가 많은 순서로 20명만 표시했습니다."
            )


# --------------------------------------------------
# 13. 혼동행렬과 목록 숫자 비교
# --------------------------------------------------

st.divider()

st.info(
    f"목록 인원 확인: 놓친 환자 {missed_count:,}명 "
    f"= 혼동행렬 FN {selected_scores['FN']:,}명 "
    f"| 헛짚은 사람 {false_alarm_count:,}명 "
    f"= 혼동행렬 FP {selected_scores['FP']:,}명"
)

st.caption(
    "※ 이 페이지는 학습용 데이터 분석 실습을 위한 것입니다. "
    "실제 의료적 진단을 대신하지 않습니다."
)
