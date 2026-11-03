
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 성능 높이기")

st.markdown("""
뇌졸중 환자는 전체 데이터에서 매우 드뭅니다.

이 페이지에서는 두 모델의 가중치를 조정하기 전과 후를 비교합니다.
가중치를 조정하면 뇌졸중 환자를 더 잘 찾아낼 수 있을까요?
""")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


# --------------------------------------------------
# 2. 입력 속성 선택
# --------------------------------------------------

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수(BMI)",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

ALL_FEATURES = list(FEATURE_LABELS.keys())

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

st.subheader("① 입력 속성 선택")

selected_features = st.multiselect(
    "모델이 뇌졸중을 예측할 때 사용할 속성을 선택하세요.",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:
    st.warning("속성을 두 가지 이상 선택해 주세요.")
    st.stop()

st.caption(
    "선택한 속성: "
    + ", ".join(FEATURE_LABELS[x] for x in selected_features)
)


# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_columns = [
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "stroke"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # BMI 이외의 필수 항목에 결측값이 있는 행 제거
    data = data.dropna(
        subset=[
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease",
            "stroke"
        ]
    )

    # id 순서대로 정렬
    data = data.sort_values("id").reset_index(drop=True)

    return data


try:
    data = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 4. 훈련용 / 테스트용 데이터 나누기
# --------------------------------------------------

# 10개씩 묶어서 앞의 3개는 테스트용,
# 나머지 7개는 훈련용으로 사용
test_mask = np.arange(len(data)) % 10 < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

X_train = train_data[selected_features].copy()
X_test = test_data[selected_features].copy()

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)


# --------------------------------------------------
# 5. BMI 결측값 처리
# --------------------------------------------------

# 중앙값은 반드시 훈련 데이터에서만 계산
train_medians = X_train.median()

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

# 나머지 속성의 결측값도 훈련용 중앙값으로 처리
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# 시각화에서 다른 속성을 고정할 때 사용할 중앙값
train_medians = X_train.median()


# --------------------------------------------------
# 6. 데이터 현황
# --------------------------------------------------

st.subheader("② 데이터 확인")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 데이터", f"{len(data):,}명")
col2.metric("훈련 데이터", f"{len(train_data):,}명")
col3.metric("테스트 데이터", f"{len(test_data):,}명")
col4.metric(
    "테스트용 뇌졸중 환자",
    f"{int(y_test.sum()):,}명"
)

st.caption(
    "뇌졸중 여부: 1은 뇌졸중, 0은 뇌졸중 아님. "
    "테스트 데이터는 두 모델 모두 동일하게 사용합니다."
)


# --------------------------------------------------
# 7. 모델 만들기
# --------------------------------------------------

st.subheader("③ 가중치 조정 전후 모델 비교")

# 가중치 조정 전 로지스틱 회귀
logistic_before = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

# 가중치 조정 후 로지스틱 회귀
logistic_after = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# 가중치 조정 전 의사결정트리
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

# 가중치 조정 후 의사결정트리
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)


# 훈련
logistic_before.fit(X_train, y_train)
logistic_after.fit(X_train, y_train)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# --------------------------------------------------
# 8. 예측 및 평가
# --------------------------------------------------

models = {
    "logistic": {
        "before": logistic_before,
        "after": logistic_after
    },
    "tree": {
        "before": tree_before,
        "after": tree_after
    }
}


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "정확도": accuracy_score(y, predictions),
        "정밀도": precision_score(
            y, predictions, zero_division=0
        ),
        "재현율": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 점수": f1_score(
            y, predictions, zero_division=0
        )
    }

    return metrics


results = {}

for model_key, model_pair in models.items():
    results[model_key] = {
        "before": evaluate_model(
            model_pair["before"], X_test, y_test
        ),
        "after": evaluate_model(
            model_pair["after"], X_test, y_test
        )
    }


# --------------------------------------------------
# 9. 비교 표 만들기
# --------------------------------------------------

def make_comparison_table(before, after):

    rows = [
        ("참음성 (TN)", "TN"),
        ("거짓양성 (FP)", "FP"),
        ("거짓음성 (FN)", "FN"),
        ("참양성 (TP)", "TP"),
        ("정확도", "정확도"),
        ("정밀도", "정밀도"),
        ("재현율", "재현율"),
        ("F1 점수", "F1 점수")
    ]

    table_data = []

    for label, key in rows:
        if key in ["TN", "FP", "FN", "TP"]:
            before_value = f"{before[key]:,}명"
            after_value = f"{after[key]:,}명"
        else:
            before_value = f"{before[key]:.3f}"
            after_value = f"{after[key]:.3f}"

        table_data.append({
            "평가 항목": label,
            "가중치 조정 전": before_value,
            "가중치 조정 후": after_value
        })

    return pd.DataFrame(table_data)


def show_model_comparison(title, before, after):

    st.markdown(f"### {title}")

    table = make_comparison_table(before, after)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    # 환자 수 비교
    tp_before = before["TP"]
    tp_after = after["TP"]

    fn_before = before["FN"]
    fn_after = after["FN"]

    fp_before = before["FP"]
    fp_after = after["FP"]

    st.markdown(
        f"**찾아낸 환자 수:** {tp_before:,}명 → {tp_after:,}명"
        f"　|　"
        f"**놓친 환자 수:** {fn_before:,}명 → {fn_after:,}명"
        f"　|　"
        f"**헛짚은 사람 수:** {fp_before:,}명 → {fp_after:,}명"
    )


col1, col2 = st.columns(2)

with col1:
    show_model_comparison(
        "로지스틱 회귀(확률로 답하는 모델)",
        results["logistic"]["before"],
        results["logistic"]["after"]
    )

with col2:
    show_model_comparison(
        "의사결정트리(질문으로 답하는 모델)",
        results["tree"]["before"],
        results["tree"]["after"]
    )


st.info("""
**표 읽는 방법**

- 참양성(TP): 실제 뇌졸중 환자를 환자로 찾아낸 사람
- 거짓음성(FN): 실제 뇌졸중 환자를 놓친 사람
- 거짓양성(FP): 뇌졸중이 아닌데 환자라고 예측한 사람
- 참음성(TN): 뇌졸중이 아닌 사람을 정확하게 예측한 사람

정확도만 보는 것보다 실제 환자를 얼마나 찾아냈는지 보여 주는
재현율도 중요합니다.
""")


# --------------------------------------------------
# 10. 경계선 시각화용 함수
# --------------------------------------------------

def get_logistic_probability(model, x_values, y_values,
                             x_feature, y_feature):

    # 나머지 속성은 훈련 데이터 중앙값으로 고정
    grid = pd.DataFrame(
        np.tile(
            train_medians[selected_features].values,
            (len(x_values) * len(y_values), 1)
        ),
        columns=selected_features
    )

    xx, yy = np.meshgrid(x_values, y_values)

    grid[x_feature] = xx.ravel()
    grid[y_feature] = yy.ravel()

    probabilities = model.predict_proba(grid)[:, 1]

    return probabilities.reshape(xx.shape)


def get_logistic_equation(model):

    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["model"]

    coefficients = (
        classifier.coef_[0] / scaler.scale_
    )

    intercept = (
        classifier.intercept_[0]
        - np.sum(
            classifier.coef_[0]
            * scaler.mean_
            / scaler.scale_
        )
    )

    return coefficients, intercept


# --------------------------------------------------
# 11. 2차원 산점도
# --------------------------------------------------

st.divider()
st.subheader("④ 실제 뇌졸중 여부와 0.5 경계선 비교")

st.markdown("""
테스트 데이터의 실제 뇌졸중 여부를 점의 색으로 표시합니다.

로지스틱 회귀 모델의 예측 확률이 0.5가 되는 경계선을
가중치 조정 전후로 겹쳐서 비교합니다.
""")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        format_func=lambda x: FEATURE_LABELS[x],
        index=0
    )

with axis_col2:
    available_y = [
        f for f in selected_features
        if f != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=available_y,
        format_func=lambda x: FEATURE_LABELS[x],
        index=0
    )


# 축 범위 설정
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

# 같은 값만 있는 경우 범위 확장
if x_min == x_max:
    x_min -= 0.5
    x_max += 0.5

if y_min == y_max:
    y_min -= 0.5
    y_max += 0.5

x_values = np.linspace(x_min, x_max, 100)
y_values = np.linspace(y_min, y_max, 100)

prob_before = get_logistic_probability(
    logistic_before,
    x_values,
    y_values,
    x_feature,
    y_feature
)

prob_after = get_logistic_probability(
    logistic_after,
    x_values,
    y_values,
    x_feature,
    y_feature
)


fig2d = go.Figure()


# 실제 뇌졸중이 아닌 사람
negative = y_test == 0

fig2d.add_trace(
    go.Scatter(
        x=X_test.loc[negative, x_feature],
        y=X_test.loc[negative, y_feature],
        mode="markers",
        name="실제 뇌졸중 아님",
        marker=dict(
            color="#3498db",
            size=7,
            opacity=0.6
        ),
        hovertemplate=(
            "가로축: %{x}<br>"
            "세로축: %{y}<br>"
            "실제 결과: 뇌졸중 아님"
            "<extra></extra>"
        )
    )
)


# 실제 뇌졸중 환자
positive = y_test == 1

fig2d.add_trace(
    go.Scatter(
        x=X_test.loc[positive, x_feature],
        y=X_test.loc[positive, y_feature],
        mode="markers",
        name="실제 뇌졸중",
        marker=dict(
            color="#e74c3c",
            size=9,
            opacity=0.85,
            symbol="circle"
        ),
        hovertemplate=(
            "가로축: %{x}<br>"
            "세로축: %{y}<br>"
            "실제 결과: 뇌졸중"
            "<extra></extra>"
        )
    )
)


# 0.5 경계선 표시
def add_boundary_2d(fig, probabilities, name, color):

    min_prob = float(np.min(probabilities))
    max_prob = float(np.max(probabilities))

    if min_prob <= 0.5 <= max_prob:
        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=probabilities,
                name=name,
                showscale=False,
                showlegend=True,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=0.01,
                    coloring="none"
                ),
                line=dict(
                    color=color,
                    width=3,
                    dash="dash"
                ),
                hoverinfo="skip"
            )
        )
        return True

    return False


boundary_before = add_boundary_2d(
    fig2d,
    prob_before,
    "가중치 조정 전 0.5 경계선",
    "#f39c12"
)

boundary_after = add_boundary_2d(
    fig2d,
    prob_after,
    "가중치 조정 후 0.5 경계선",
    "#8e44ad"
)


fig2d.update_layout(
    title="실제 결과와 로지스틱 회귀의 0.5 경계선",
    xaxis_title=FEATURE_LABELS[x_feature],
    yaxis_title=FEATURE_LABELS[y_feature],
    height=650,
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig2d, use_container_width=True)


if not boundary_before:
    st.warning(
        "가중치 조정 전의 0.5 경계선은 현재 그림의 범위 밖에 있습니다."
    )

if not boundary_after:
    st.warning(
        "가중치 조정 후의 0.5 경계선은 현재 그림의 범위 밖에 있습니다."
    )

st.caption("""
주황색 점선: 가중치 조정 전
보라색 점선: 가중치 조정 후
파란색 점: 실제 뇌졸중 아님
빨간색 점: 실제 뇌졸중
""")


# --------------------------------------------------
# 12. 3차원 산점도 및 0.5 평면
# --------------------------------------------------

if len(selected_features) >= 3:

    st.divider()
    st.subheader("⑤ 3차원 산점도와 0.5 경계 평면")

    st.markdown("""
    속성을 세 가지 골라 3차원 공간에서 살펴봅니다.

    마우스로 그림을 드래그하면 여러 방향에서 회전하여 볼 수 있습니다.
    """)
    
    axis_3d = st.multiselect(
        "3차원 그래프의 축으로 사용할 속성 세 가지를 선택하세요.",
        options=selected_features,
        default=selected_features[:3],
        max_selections=3,
        format_func=lambda x: FEATURE_LABELS[x],
        key="axis_3d"
    )

    if len(axis_3d) == 3:

        x3, y3, z3 = axis_3d

        fig3d = go.Figure()

        # 로지스틱 회귀의 실제 계수와 절편
        coef_before, intercept_before = get_logistic_equation(
            logistic_before
        )

        coef_after, intercept_after = get_logistic_equation(
            logistic_after
        )

        # 두 평면을 그리기 위한 x, y 격자
        x3_min = float(X_test[x3].min())
        x3_max = float(X_test[x3].max())

        y3_min = float(X_test[y3].min())
        y3_max = float(X_test[y3].max())

        z3_min = float(X_test[z3].min())
        z3_max = float(X_test[z3].max())

        if x3_min == x3_max:
            x3_min -= 0.5
            x3_max += 0.5

        if y3_min == y3_max:
            y3_min -= 0.5
            y3_max += 0.5

        if z3_min == z3_max:
            z3_min -= 0.5
            z3_max += 0.5

        gx = np.linspace(x3_min, x3_max, 35)
        gy = np.linspace(y3_min, y3_max, 35)

        xx3, yy3 = np.meshgrid(gx, gy)

        # 선택하지 않은 속성은 훈련 데이터 중앙값으로 고정
        def calculate_plane(coefficients, intercept):

            ix = selected_features.index(x3)
            iy = selected_features.index(y3)
            iz = selected_features.index(z3)

            ax = coefficients[ix]
            ay = coefficients[iy]
            az = coefficients[iz]

            # 선택한 세 속성 이외의 항목을 상수로 계산
            constant = intercept

            for i, feature in enumerate(selected_features):
                if feature not in [x3, y3, z3]:
                    constant += (
                        coefficients[i]
                        * train_medians[feature]
                    )

            if abs(az) < 1e-10:
                return None, False

            zz3 = -(
                constant
                + ax * xx3
                + ay * yy3
            ) / az

            # 그림의 z축 범위 안에 들어오는 부분만 표시
            visible = (
                (zz3 >= z3_min)
                & (zz3 <= z3_max)
            )

            if not np.any(visible):
                return None, False

            zz_visible = np.where(visible, zz3, np.nan)

            return zz_visible, True


        plane_before, visible_before = calculate_plane(
            coef_before,
            intercept_before
        )

        plane_after, visible_after = calculate_plane(
            coef_after,
            intercept_after
        )


        # 평면을 먼저 그려서 점이 잘 보이도록 함
        if visible_before:
            fig3d.add_trace(
                go.Surface(
                    x=xx3,
                    y=yy3,
                    z=plane_before,
                    name="가중치 조정 전 0.5 평면",
                    opacity=0.45,
                    colorscale=[
                        [0, "#f39c12"],
                        [1, "#f39c12"]
                    ],
                    showscale=False,
                    connectgaps=False,
                    hoverinfo="skip"
                )
            )

        if visible_after:
            fig3d.add_trace(
                go.Surface(
                    x=xx3,
                    y=yy3,
                    z=plane_after,
                    name="가중치 조정 후 0.5 평면",
                    opacity=0.45,
                    colorscale=[
                        [0, "#8e44ad"],
                        [1, "#8e44ad"]
                    ],
                    showscale=False,
                    connectgaps=False,
                    hoverinfo="skip"
                )
            )


        # 실제 뇌졸중 여부에 따라 점 표시
        for label, color, name in [
            (0, "#3498db", "실제 뇌졸중 아님"),
            (1, "#e74c3c", "실제 뇌졸중")
        ]:

            mask = y_test == label

            fig3d.add_trace(
                go.Scatter3d(
                    x=X_test.loc[mask, x3],
                    y=X_test.loc[mask, y3],
                    z=X_test.loc[mask, z3],
                    mode="markers",
                    name=name,
                    marker=dict(
                        size=4 if label == 0 else 5,
                        color=color,
                        opacity=0.7
                    )
                )
            )


        fig3d.update_layout(
            title="실제 뇌졸중 여부와 로지스틱 회귀의 0.5 평면",
            height=750,
            scene=dict(
                xaxis_title=FEATURE_LABELS[x3],
                yaxis_title=FEATURE_LABELS[y3],
                zaxis_title=FEATURE_LABELS[z3],
                aspectmode="auto"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        st.plotly_chart(
            fig3d,
            use_container_width=True
        )

        if not visible_before:
            st.warning(
                "가중치 조정 전의 0.5 평면은 현재 3차원 그림의 범위 밖에 있습니다."
            )

        if not visible_after:
            st.warning(
                "가중치 조정 후의 0.5 평면은 현재 3차원 그림의 범위 밖에 있습니다."
            )

        if visible_before or visible_after:
            st.caption("""
            주황색 평면: 가중치 조정 전
            보라색 평면: 가중치 조정 후
            파란색 점: 실제 뇌졸중 아님
            빨간색 점: 실제 뇌졸중
            """)

    else:
        st.info("3차원 그래프를 보려면 속성을 세 가지 선택하세요.")


# --------------------------------------------------
# 13. 마무리
# --------------------------------------------------

st.divider()

st.subheader("⑥ 생각해 보기")

st.markdown("""
다음 질문을 생각해 보세요.

1. 가중치를 조정한 뒤 실제 뇌졸중 환자를 더 많이 찾아냈나요?
2. 환자를 더 많이 찾아내는 대신 헛짚은 사람도 늘어났나요?
3. 정확도와 재현율 중 어떤 지표가 이 문제에서 더 중요할까요?
4. 가중치를 조정하는 것이 모든 모델에서 똑같은 효과를 냈나요?

**가중치 조정의 핵심은 단순히 정확도를 높이는 것이 아니라,
드문 양성 사례를 놓치지 않도록 모델의 판단을 조정하는 것입니다.**
""")
