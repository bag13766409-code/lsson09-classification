import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# ==================================================
# 1. 기본 설정
# ==================================================

st.title("🚀 성능 높이기")

st.write(
    "뇌졸중 환자는 전체 사람 중 아주 적습니다. "
    "두 종류의 사람을 똑같이 취급하는 대신, "
    "뇌졸중 환자에게 더 큰 가중치를 주면 "
    "결과가 어떻게 달라지는지 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

MODEL_NAMES = {
    "logistic": "로지스틱 회귀(확률로 답하는 모델)",
    "tree": "의사결정트리(질문으로 답하는 모델)",
}


# ==================================================
# 2. 데이터 읽기
# ==================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke",
    ]

    for col in required_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease",
            "stroke",
        ]
    ).copy()

    data = data.sort_values(
        "id",
        kind="stable"
    ).reset_index(drop=True)

    return data


# ==================================================
# 3. 데이터 나누기
# ==================================================

def split_data(data):

    test_mask = np.arange(len(data)) % 10 < 3

    train_data = data.loc[~test_mask].copy()
    test_data = data.loc[test_mask].copy()

    return train_data, test_data


# ==================================================
# 4. 평가 지표 계산
# ==================================================

def calculate_scores(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0
    )

    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0 else None
    )

    predicted_positive = tp + fp

    precision = (
        tp / predicted_positive
        if predicted_positive > 0 else None
    )

    if (
        precision is None
        or recall is None
        or precision + recall == 0
    ):
        f1 = None
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

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


# ==================================================
# 5. 모델 학습
# ==================================================

def train_models(X_train, X_test, y_train):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # BMI 결측값은 훈련용 중앙값으로 채우기
    if "bmi" in X_train.columns:

        bmi_median = X_train["bmi"].median()

        if pd.isna(bmi_median):
            raise ValueError(
                "훈련용 체질량지수에 유효한 값이 없습니다."
            )

        X_train["bmi"] = X_train["bmi"].fillna(
            bmi_median
        )

        X_test["bmi"] = X_test["bmi"].fillna(
            bmi_median
        )

    # 로지스틱 회귀용 표준화
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # --------------------------------------------------
    # 로지스틱 회귀: 가중치 조정 전
    # --------------------------------------------------

    logistic_before = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    logistic_before.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------
    # 로지스틱 회귀: 가중치 조정 후
    # --------------------------------------------------

    logistic_after = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    logistic_after.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------
    # 의사결정트리: 가중치 조정 전
    # --------------------------------------------------

    tree_before = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )

    tree_before.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 의사결정트리: 가중치 조정 후
    # --------------------------------------------------

    tree_after = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    tree_after.fit(
        X_train,
        y_train
    )

    models = {
        "logistic": {
            "before": logistic_before,
            "after": logistic_after,
        },
        "tree": {
            "before": tree_before,
            "after": tree_after,
        }
    }

    return models, scaler, X_train, X_test


# ==================================================
# 6. 확률 예측
# ==================================================

def get_probability(
    model,
    X,
    model_type,
    scaler
):

    if model_type == "logistic":
        X_input = scaler.transform(X)
    else:
        X_input = X

    probabilities = model.predict_proba(X_input)

    positive_index = list(model.classes_).index(1)

    return probabilities[:, positive_index]


# ==================================================
# 7. 비교 표 만들기
# ==================================================

def make_comparison_table(before, after):

    rows = [
        ("TN: 제대로 맞힌 음성", "TN"),
        ("FP: 헛짚은 사람", "FP"),
        ("FN: 놓친 환자", "FN"),
        ("TP: 찾아낸 환자", "TP"),
        ("정확도", "정확도"),
        ("재현율", "재현율"),
        ("정밀도", "정밀도"),
        ("F1", "F1"),
    ]

    result = []

    for label, key in rows:

        if key in ["TN", "FP", "FN", "TP"]:

            before_value = f"{before[key]:,}명"
            after_value = f"{after[key]:,}명"

        else:

            before_value = format_score(before[key])
            after_value = format_score(after[key])

        result.append({
            "평가 항목": label,
            "가중치 조정 전": before_value,
            "가중치 조정 후": after_value,
        })

    return pd.DataFrame(result)


def show_change(before, after):

    st.info(
        f"찾아낸 환자(TP): {before['TP']:,}명 → {after['TP']:,}명"
        f"  |  놓친 환자(FN): {before['FN']:,}명 → {after['FN']:,}명"
        f"  |  헛짚은 사람(FP): {before['FP']:,}명 → {after['FP']:,}명"
    )


# ==================================================
# 8. 2차원 산점도와 경계선
# ==================================================

def make_2d_chart(
    X_test,
    y_test,
    models,
    scaler,
    features,
    model_type
):

    x_feature, y_feature = features

    x_min = X_test[x_feature].min()
    x_max = X_test[x_feature].max()

    y_min = X_test[y_feature].min()
    y_max = X_test[y_feature].max()

    x_padding = max((x_max - x_min) * 0.08, 0.1)
    y_padding = max((y_max - y_min) * 0.08, 0.1)

    x_values = np.linspace(
        x_min - x_padding,
        x_max + x_padding,
        100
    )

    y_values = np.linspace(
        y_min - y_padding,
        y_max + y_padding,
        100
    )

    xx, yy = np.meshgrid(
        x_values,
        y_values
    )

    # 다른 속성은 테스트 데이터 중앙값으로 고정
    grid = pd.DataFrame(
        np.tile(
            X_test.median().values,
            (xx.size, 1)
        ),
        columns=X_test.columns
    )

    grid[x_feature] = xx.ravel()
    grid[y_feature] = yy.ravel()

    grid_before = get_probability(
        models["before"],
        grid,
        model_type,
        scaler
    ).reshape(xx.shape)

    grid_after = get_probability(
        models["after"],
        grid,
        model_type,
        scaler
    ).reshape(xx.shape)

    fig = go.Figure()

    # 실제 뇌졸중이 아닌 사람
    negative = y_test == 0

    fig.add_trace(
        go.Scatter(
            x=X_test.loc[negative, x_feature],
            y=X_test.loc[negative, y_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#3498DB",
                size=6,
                opacity=0.55
            )
        )
    )

    # 실제 뇌졸중인 사람
    positive = y_test == 1

    fig.add_trace(
        go.Scatter(
            x=X_test.loc[positive, x_feature],
            y=X_test.loc[positive, y_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E74C3C",
                size=8,
                opacity=0.85,
                symbol="diamond"
            )
        )
    )

    # 경계선이 그림 안에 있는지 확인
    before_inside = (
        np.min(grid_before) <= 0.5
        <= np.max(grid_before)
    )

    after_inside = (
        np.min(grid_after) <= 0.5
        <= np.max(grid_after)
    )

    # 가중치 조정 전 경계선
    if before_inside:

        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=grid_before,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1
                ),
                showscale=False,
                name="가중치 조정 전 0.5 경계선",
                line=dict(
                    color="#E67E22",
                    width=4
                ),
                hoverinfo="skip"
            )
        )

    # 가중치 조정 후 경계선
    if after_inside:

        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=grid_after,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=1
                ),
                showscale=False,
                name="가중치 조정 후 0.5 경계선",
                line=dict(
                    color="#16A085",
                    width=4,
                    dash="dash"
                ),
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title="실제 뇌졸중 여부와 0.5 경계선",
        xaxis_title=FEATURE_LABELS[x_feature],
        yaxis_title=FEATURE_LABELS[y_feature],
        legend_title="범례",
        height=650,
        template="plotly_white"
    )

    return fig, before_inside, after_inside


# ==================================================
# 9. 3차원 산점도와 0.5 경계면
# ==================================================

def make_3d_chart(
    X_test,
    y_test,
    models,
    scaler,
    features,
    model_type
):

    x_feature, y_feature, z_feature = features

    # 그림의 범위
    x_min = X_test[x_feature].min()
    x_max = X_test[x_feature].max()

    y_min = X_test[y_feature].min()
    y_max = X_test[y_feature].max()

    z_min = X_test[z_feature].min()
    z_max = X_test[z_feature].max()

    def make_range(minimum, maximum):

        padding = max(
            (maximum - minimum) * 0.05,
            0.1
        )

        return np.linspace(
            minimum - padding,
            maximum + padding,
            22
        )

    x_values = make_range(x_min, x_max)
    y_values = make_range(y_min, y_max)
    z_values = make_range(z_min, z_max)

    xx, yy, zz = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij"
    )

    # 선택하지 않은 나머지 속성은 중앙값으로 고정
    grid = pd.DataFrame(
        np.tile(
            X_test.median().values,
            (xx.size, 1)
        ),
        columns=X_test.columns
    )

    grid[x_feature] = xx.ravel()
    grid[y_feature] = yy.ravel()
    grid[z_feature] = zz.ravel()

    before_probability = get_probability(
        models["before"],
        grid,
        model_type,
        scaler
    )

    after_probability = get_probability(
        models["after"],
        grid,
        model_type,
        scaler
    )

    before_inside = (
        np.min(before_probability) <= 0.5
        <= np.max(before_probability)
    )

    after_inside = (
        np.min(after_probability) <= 0.5
        <= np.max(after_probability)
    )

    fig = go.Figure()

    # 실제 뇌졸중이 아닌 사람
    negative = y_test == 0

    fig.add_trace(
        go.Scatter3d(
            x=X_test.loc[negative, x_feature],
            y=X_test.loc[negative, y_feature],
            z=X_test.loc[negative, z_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#3498DB",
                size=4,
                opacity=0.5
            )
        )
    )

    # 실제 뇌졸중인 사람
    positive = y_test == 1

    fig.add_trace(
        go.Scatter3d(
            x=X_test.loc[positive, x_feature],
            y=X_test.loc[positive, y_feature],
            z=X_test.loc[positive, z_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E74C3C",
                size=5,
                opacity=0.9,
                symbol="diamond"
            )
        )
    )

    # 가중치 조정 전 0.5 경계면
    if before_inside:

        fig.add_trace(
            go.Isosurface(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=before_probability,
                isomin=0.5,
                isomax=0.5,
                surface_count=1,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                colorscale=[
                    [0, "#E67E22"],
                    [1, "#E67E22"]
                ],
                showscale=False,
                opacity=0.45,
                name="가중치 조정 전 0.5 경계면",
                hoverinfo="skip"
            )
        )

    # 가중치 조정 후 0.5 경계면
    if after_inside:

        fig.add_trace(
            go.Isosurface(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=after_probability,
                isomin=0.5,
                isomax=0.5,
                surface_count=1,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                colorscale=[
                    [0, "#16A085"],
                    [1, "#16A085"]
                ],
                showscale=False,
                opacity=0.45,
                name="가중치 조정 후 0.5 경계면",
                hoverinfo="skip"
            )
        )

    fig.update_layout(
        title="3차원 산점도와 0.5 경계면",
        scene=dict(
            xaxis_title=FEATURE_LABELS[x_feature],
            yaxis_title=FEATURE_LABELS[y_feature],
            zaxis_title=FEATURE_LABELS[z_feature],
        ),
        legend_title="범례",
        height=750,
        template="plotly_white"
    )

    return fig, before_inside, after_inside


# ==================================================
# 10. 화면: 속성 선택
# ==================================================

st.divider()
st.subheader("1. 입력으로 사용할 속성 선택")

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=list(FEATURE_LABELS.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:

    st.warning(
        "산점도를 그리려면 속성을 최소 2개 선택해야 합니다."
    )

    st.stop()


# ==================================================
# 11. 데이터 준비
# ==================================================

try:

    data = load_data()

    train_data, test_data = split_data(data)

    X_train = train_data[selected_features].copy()
    X_test = test_data[selected_features].copy()

    y_train = train_data["stroke"].astype(int)
    y_test = test_data["stroke"].astype(int)

    if y_train.nunique() < 2:

        st.error(
            "훈련용 데이터에 뇌졸중 여부 두 종류가 모두 있어야 합니다."
        )

        st.stop()

    models, scaler, X_train, X_test = train_models(
        X_train,
        X_test,
        y_train
    )

except Exception as e:

    st.error(f"데이터 처리 또는 모델 학습 오류: {e}")

    st.stop()


# ==================================================
# 12. 테스트 데이터 정보
# ==================================================

st.divider()
st.subheader("2. 테스트 데이터")

test_positive_count = int((y_test == 1).sum())

st.info(
    f"테스트용 사람 수: **{len(test_data):,}명** "
    f"| 실제 뇌졸중 환자: **{test_positive_count:,}명** "
    f"| 훈련용 사람 수: **{len(train_data):,}명**"
)


# ==================================================
# 13. 두 모델의 예측
# ==================================================

results = {}

for model_type in ["logistic", "tree"]:

    model_pair = models[model_type]

    before_probability = get_probability(
        model_pair["before"],
        X_test,
        model_type,
        scaler
    )

    after_probability = get_probability(
        model_pair["after"],
        X_test,
        model_type,
        scaler
    )

    before_pred = (
        before_probability >= 0.5
    ).astype(int)

    after_pred = (
        after_probability >= 0.5
    ).astype(int)

    before_scores = calculate_scores(
        y_test,
        before_pred
    )

    after_scores = calculate_scores(
        y_test,
        after_pred
    )

    results[model_type] = {
        "before_pred": before_pred,
        "after_pred": after_pred,
        "before_scores": before_scores,
        "after_scores": after_scores,
    }


# ==================================================
# 14. 로지스틱 회귀 비교
# ==================================================

st.divider()
st.subheader("3. 로지스틱 회귀(확률로 답하는 모델)")

st.write(
    "가중치를 조정하기 전과 후의 혼동행렬과 "
    "평가 지표를 비교합니다."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("**가중치 조정 전**")

    st.table(
        make_comparison_table(
            results["logistic"]["before_scores"],
            results["logistic"]["before_scores"]
        ).iloc[:, :2]
    )

with col2:

    st.markdown("**가중치 조정 후**")

    st.table(
        make_comparison_table(
            results["logistic"]["before_scores"],
            results["logistic"]["after_scores"]
        ).iloc[:, [0, 2]].rename(
            columns={"가중치 조정 후": "가중치 조정 후"}
        )
    )

show_change(
    results["logistic"]["before_scores"],
    results["logistic"]["after_scores"]
)


# ==================================================
# 15. 의사결정트리 비교
# ==================================================

st.divider()
st.subheader("4. 의사결정트리(질문으로 답하는 모델)")

st.write(
    "가중치를 조정하기 전과 후의 혼동행렬과 "
    "평가 지표를 비교합니다."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("**가중치 조정 전**")

    st.table(
        make_comparison_table(
            results["tree"]["before_scores"],
            results["tree"]["before_scores"]
        ).iloc[:, :2]
    )

with col2:

    st.markdown("**가중치 조정 후**")

    st.table(
        make_comparison_table(
            results["tree"]["before_scores"],
            results["tree"]["after_scores"]
        ).iloc[:, [0, 2]]
    )

show_change(
    results["tree"]["before_scores"],
    results["tree"]["after_scores"]
)


# ==================================================
# 16. 산점도에 사용할 축 선택
# ==================================================

st.divider()
st.subheader("5. 산점도와 0.5 경계선")

st.write(
    "테스트용 사람들을 실제 뇌졸중 여부에 따라 색으로 구분합니다. "
    "주황색은 가중치 조정 전 경계선, "
    "초록색은 가중치 조정 후 경계선입니다."
)

if len(selected_features) == 2:

    st.markdown("**2차원 산점도**")

    col1, col2 = st.columns(2)

    with col1:

        x_feature = st.selectbox(
            "가로축 속성",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x]
        )

    remaining_features = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    with col2:

        y_feature = st.selectbox(
            "세로축 속성",
            remaining_features,
            format_func=lambda x: FEATURE_LABELS[x]
        )

    selected_axes = [x_feature, y_feature]

else:

    st.markdown("**3차원 산점도**")

    col1, col2, col3 = st.columns(3)

    with col1:

        x_feature = st.selectbox(
            "가로축 속성",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x]
        )

    remaining_1 = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    with col2:

        y_feature = st.selectbox(
            "세로축 속성",
            remaining_1,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x]
        )

    remaining_2 = [
        feature
        for feature in remaining_1
        if feature != y_feature
    ]

    with col3:

        z_feature = st.selectbox(
            "깊이축 속성",
            remaining_2,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x]
        )

    selected_axes = [
        x_feature,
        y_feature,
        z_feature
    ]


# ==================================================
# 17. 모델별 산점도
# ==================================================

model_choice = st.selectbox(
    "산점도로 확인할 모델",
    options=["logistic", "tree"],
    format_func=lambda x: MODEL_NAMES[x]
)

model_pair = models[model_choice]

if len(selected_axes) == 2:

    fig, before_inside, after_inside = make_2d_chart(
        X_test,
        y_test,
        model_pair,
        scaler,
        selected_axes,
        model_choice
    )

else:

    fig, before_inside, after_inside = make_3d_chart(
        X_test,
        y_test,
        model_pair,
        scaler,
        selected_axes,
        model_choice
    )


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 18. 경계선이 그림 밖에 있는지 안내
# ==================================================

if not before_inside:

    st.warning(
        "가중치 조정 전의 0.5 경계선은 "
        "현재 그림의 범위 밖에 있습니다."
    )

if not after_inside:

    st.warning(
        "가중치 조정 후의 0.5 경계선은 "
        "현재 그림의 범위 밖에 있습니다."
    )


# ==================================================
# 19. 설명
# ==================================================

st.divider()
st.subheader("6. 결과 이해하기")

st.markdown(
    """
    **가중치 조정이란?**

    뇌졸중을 겪은 사람은 전체 데이터에서 차지하는 비율이 매우 낮습니다.

    그래서 모델이 모든 사람을 뇌졸중이 아니라고 예측해도
    정확도가 높게 나올 수 있습니다.

    `class_weight="balanced"`를 사용하면
    적은 수의 뇌졸중 환자를 틀렸을 때 더 큰 벌점을 주도록 학습합니다.

    **결과를 살펴볼 때 주의할 점**

    - 찾아낸 환자(TP)가 늘어나면 실제 환자를 더 많이 찾아낸 것입니다.
    - 놓친 환자(FN)가 줄어들면 실제 환자를 놓치는 경우가 줄어든 것입니다.
    - 헛짚은 사람(FP)이 늘어나면 실제 환자가 아닌 사람을 환자로 예측하는 경우가 늘어난 것입니다.
    - 재현율이 높아져도 정밀도나 정확도가 낮아질 수 있습니다.
    - 가중치를 조정했다고 해서 모든 평가 지표가 좋아지는 것은 아닙니다.

    **0.5 경계선이란?**

    모델이 뇌졸중일 확률을 계산했을 때,
    확률이 0.5 이상이면 뇌졸중으로 예측하고
    0.5 미만이면 뇌졸중이 아닌 것으로 예측합니다.

    그래프의 경계선은 이 두 예측이 나뉘는 위치입니다.

    ※ 이 페이지는 머신러닝 학습 실습용이며,
    실제 의료 진단을 대신하지 않습니다.
    """
)</WritingBlock>

**파일 위치**

```text
프로젝트/
├── main.py
├── requirements.txt
└── pages/
    ├── 2_분류_모델.py
    ├── 3_채점표.py
    └── 4_성능_높이기.py
