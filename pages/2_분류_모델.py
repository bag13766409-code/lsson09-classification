
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# =====================================================
# 1. 기본 설정
# =====================================================
st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧪",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

RANDOM_STATE = 42

# 우리말 열 이름
KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

# 기본 선택 속성
DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

ALL_FEATURES = list(KOREAN_NAMES.keys())


# =====================================================
# 2. 데이터 불러오기
# =====================================================
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


# 필요한 열 확인
required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


# =====================================================
# 3. 데이터 정리와 테스트 데이터 분리
# =====================================================
data = df[required_columns].copy()

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
        "stroke"
    ]
)

data = data.sort_values(
    "id",
    kind="stable"
).reset_index(drop=True)

# 정답은 0 또는 1이어야 함
data = data[data["stroke"].isin([0, 1])].reset_index(drop=True)

# 10명씩 묶어서 앞의 3명은 테스트용
# 나머지 7명은 훈련용
test_mask = np.arange(len(data)) % 10 < 3

train_df = data.loc[~test_mask].copy()
test_df = data.loc[test_mask].copy()

# 정답
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 테스트 인원 확인
if len(test_df) != 1533:
    st.error(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 테스트 인원: {len(test_df):,}명"
    )
    st.stop()


# =====================================================
# 4. 화면 제목
# =====================================================
st.title("🧪 분류 모델")

st.write("""
선택한 건강 정보를 바탕으로 뇌졸중 여부를 예측하는
분류 모델을 만들어 봅니다.

- stroke = 1 : 뇌졸중 (양성)
- stroke = 0 : 뇌졸중 아님 (음성)

모델의 성능은 학습에 사용하지 않은 테스트 데이터로 평가합니다.
""")


# =====================================================
# 5. 입력 속성 선택
# =====================================================
st.header("1. 예측에 사용할 속성 고르기")

st.write(
    "모델이 예측할 때 사용할 건강 정보를 선택하세요. "
    "처음에는 체질량지수를 제외한 네 가지가 선택되어 있습니다."
)

selected_features = st.multiselect(
    "입력 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: KOREAN_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

st.write(
    "**선택한 속성:** "
    + ", ".join(KOREAN_NAMES[x] for x in selected_features)
)


# =====================================================
# 6. BMI 결측값 처리
# =====================================================
# 선택한 속성만 사용
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

bmi_median = None

if "bmi" in selected_features:

    # 훈련 데이터의 중앙값만 계산
    bmi_median = X_train["bmi"].median()

    # 훈련 데이터와 테스트 데이터 모두 훈련 중앙값으로 대체
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"체질량지수의 빈 값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# =====================================================
# 7. 크기 맞추기
# =====================================================
# 크기 맞추기는 훈련 데이터만 이용해 학습
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================
# 8. 로지스틱 회귀 모델
# =====================================================
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# =====================================================
# 9. 의사결정트리 모델
# =====================================================
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

# 의사결정트리는 원래 단위의 값으로 학습
tree_model.fit(
    X_train,
    y_train
)


# =====================================================
# 10. 기준 모델
# =====================================================
# 입력 속성을 하나도 보지 않고 훈련 데이터에서
# 더 많은 쪽으로만 답하는 모델
majority_class = int(y_train.mode().iloc[0])

baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)


# =====================================================
# 11. 정확도 계산
# =====================================================
logistic_train_accuracy = logistic_model.score(
    X_train_scaled,
    y_train
)

logistic_test_accuracy = logistic_model.score(
    X_test_scaled,
    y_test
)

tree_train_accuracy = tree_model.score(
    X_train,
    y_train
)

tree_test_accuracy = tree_model.score(
    X_test,
    y_test
)

baseline_train_accuracy = (
    baseline_train_pred == y_train.to_numpy()
).mean()

baseline_test_accuracy = (
    baseline_test_pred == y_test.to_numpy()
).mean()


# =====================================================
# 12. 정확도 카드
# =====================================================
st.divider()

st.header("2. 모델의 정확도 비교")

st.write("""
정확도는 전체 예측 가운데 정답을 맞힌 비율입니다.

큰 숫자는 테스트 데이터의 정확도이고,
아래에는 훈련 데이터와 테스트 데이터의 정확도를 함께 표시합니다.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "로지스틱 회귀",
        f"{logistic_test_accuracy * 100:.2f}%"
    )
    st.caption(
        "확률로 답하는 모델\n"
        f"훈련 데이터: {logistic_train_accuracy * 100:.2f}%\n\n"
        f"테스트 데이터: {logistic_test_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "의사결정트리",
        f"{tree_test_accuracy * 100:.2f}%"
    )
    st.caption(
        "질문으로 답하는 모델\n"
        f"훈련 데이터: {tree_train_accuracy * 100:.2f}%\n\n"
        f"테스트 데이터: {tree_test_accuracy * 100:.2f}%"
    )

with col3:
    st.metric(
        "입력을 보지 않는 기준 모델",
        f"{baseline_test_accuracy * 100:.2f}%"
    )
    st.caption(
        f"훈련 데이터: {baseline_train_accuracy * 100:.2f}%\n\n"
        f"테스트 데이터: {baseline_test_accuracy * 100:.2f}%"
    )

st.caption(
    f"기준 모델은 훈련 데이터에서 더 많은 정답인 "
    f"{'뇌졸중' if majority_class == 1 else '뇌졸중 아님'}만 예측합니다."
)


# =====================================================
# 13. 산점도 축 선택
# =====================================================
st.divider()

st.header("3. 두 속성으로 모델의 판단 살펴보기")

st.write("""
선택한 입력 속성 중 두 개를 골라 가로축과 세로축으로 사용합니다.

점의 색은 실제 뇌졸중 여부를 나타냅니다.
배경의 옅은 색은 의사결정트리의 예측 영역입니다.
선은 로지스틱 회귀 모델이 확률 0.5를 기준으로 나누는 경계입니다.
""")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x]
    )

with axis_col2:
    y_options = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x]
    )


# =====================================================
# 14. 나머지 속성 고정값 안내
# =====================================================
other_features = [
    feature for feature in selected_features
    if feature not in [x_feature, y_feature]
]

fixed_values = {}

for feature in other_features:
    fixed_values[feature] = float(
        X_test[feature].median()
    )

if fixed_values:
    fixed_text = ", ".join(
        f"{KOREAN_NAMES[key]} = {value:.2f}"
        for key, value in fixed_values.items()
    )

    st.info(
        "그림의 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 "
        f"고정하여 계산합니다: {fixed_text}"
    )
else:
    st.info("선택한 입력 속성이 두 개뿐이므로 고정할 나머지 속성이 없습니다.")


# =====================================================
# 15. 예측 영역을 계산할 격자 만들기
# =====================================================
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

# 범위가 같은 경우를 대비
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

if x_padding == 0:
    x_padding = 1

if y_padding == 0:
    y_padding = 1

x_values = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    120
)

y_values = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    120
)

xx, yy = np.meshgrid(
    x_values,
    y_values
)

grid_data = pd.DataFrame({
    feature: np.full(
        xx.size,
        fixed_values.get(
            feature,
            0
        )
    )
    for feature in selected_features
})

grid_data[x_feature] = xx.ravel()
grid_data[y_feature] = yy.ravel()

# 열 순서를 모델 학습 순서와 맞춤
grid_data = grid_data[selected_features]


# =====================================================
# 16. 로지스틱 회귀 확률 계산
# =====================================================
grid_scaled = scaler.transform(grid_data)

logistic_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1]

probability_grid = logistic_probability.reshape(
    xx.shape
)


# =====================================================
# 17. 의사결정트리 예측 영역 계산
# =====================================================
tree_grid_prediction = tree_model.predict(
    grid_data
)

tree_prediction_grid = tree_grid_prediction.reshape(
    xx.shape
)


# =====================================================
# 18. 테스트 데이터 산점도
# =====================================================
plot_df = test_df[
    [x_feature, y_feature, "stroke"]
].copy()

plot_df["실제 결과"] = plot_df["stroke"].map({
    0: "뇌졸중 아님",
    1: "뇌졸중"
})

plot_df["실제 결과"] = pd.Categorical(
    plot_df["실제 결과"],
    categories=["뇌졸중 아님", "뇌졸중"],
    ordered=True
)


fig = go.Figure()


# 의사결정트리 예측 영역
fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=tree_prediction_grid,
        colorscale=[
            [0, "rgba(76,120,168,0.22)"],
            [1, "rgba(228,87,86,0.22)"]
        ],
        zmin=0,
        zmax=1,
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill",
            showlines=False
        ),
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 예측 영역"
    )
)


# 로지스틱 회귀의 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=probability_grid,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlabels=False
        ),
        line=dict(
            color="#222222",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선"
    )
)


# 실제 테스트 데이터
color_map = {
    "뇌졸중 아님": "#2878B5",
    "뇌졸중": "#E45756"
}

for label in ["뇌졸중 아님", "뇌졸중"]:

    group = plot_df[
        plot_df["실제 결과"] == label
    ]

    fig.add_trace(
        go.Scatter(
            x=group[x_feature],
            y=group[y_feature],
            mode="markers",
            name=label,
            marker=dict(
                color=color_map[label],
                size=7,
                opacity=0.7
            ),
            hovertemplate=(
                KOREAN_NAMES[x_feature] + ": %{x:.2f}<br>"
                + KOREAN_NAMES[y_feature] + ": %{y:.2f}<br>"
                + "실제 결과: " + label
                + "<extra></extra>"
            )
        )
    )


fig.update_layout(
    title="테스트 데이터와 두 모델의 판단 영역",
    xaxis_title=KOREAN_NAMES[x_feature],
    yaxis_title=KOREAN_NAMES[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650,
    template="plotly_white"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =====================================================
# 19. 로지스틱 회귀 경계선이 그림 밖인지 확인
# =====================================================
# 경계선은 0.5 확률을 기준으로 함
# 격자 안에서 0.5보다 작은 값과 큰 값이 모두 있어야
# 경계선이 그림 안에 나타날 수 있음

prob_min = float(probability_grid.min())
prob_max = float(probability_grid.max())

if prob_min > 0.5 or prob_max < 0.5:
    st.warning(
        "로지스틱 회귀의 확률 0.5 경계선이 현재 그림 범위 밖에 있습니다. "
        "따라서 그림 안에 경계선이 나타나지 않습니다."
    )
else:
    st.success(
        "로지스틱 회귀의 확률 0.5 경계선이 그림 범위 안에 있습니다."
    )


# =====================================================
# 20. 의사결정트리 질문 나무
# =====================================================
st.divider()

st.header("4. 의사결정트리가 던지는 질문")

st.write("""
의사결정트리는 질문을 최대 세 번까지 던집니다.

각 마디에는 해당 마디에 도착한 훈련용 사람 수,
실제 뇌졸중인 사람 수, 뇌졸중 비율을 표시합니다.

마지막 마디는 예측 결과에 따라 색을 다르게 표시합니다.
""")


# =====================================================
# 21. DOT 문자열 만들기
# =====================================================
tree_structure = tree_model.tree_

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, nodesep=0.45, ranksep=0.65];',
    'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.15,0.10"];',
    'edge [fontname="Arial", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
asked_features = set()


def make_node(node_id):

    global leaf_count, negative_leaf_count

    samples = int(tree_structure.n_node_samples[node_id])

    # 클래스별 사람 수
    values = tree_structure.value[node_id][0]

    negative_count = int(round(values[0]))
    positive_count = int(round(values[1]))

    total_count = negative_count + positive_count

    positive_rate = (
        positive_count / total_count * 100
        if total_count > 0 else 0
    )

    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:

        leaf_count += 1

        predicted_class = int(
            tree_structure.value[node_id][0].argmax()
        )

        if predicted_class == 0:
            negative_leaf_count += 1
            answer = "예측: 뇌졸중 아님"
            fill_color = "#B9DDF4"
        else:
            answer = "예측: 뇌졸중"
            fill_color = "#F6C1C1"

        label = (
            f"마지막 마디 {node_id}\\n"
            f"훈련용 사람 수: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%\\n"
            f"{answer}"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

    else:

        feature_index = tree_structure.feature[node_id]

        feature_name = selected_features[feature_index]

        asked_features.add(feature_name)

        threshold = tree_structure.threshold[node_id]

        label = (
            f"질문 {node_id}\\n"
            f"{KOREAN_NAMES[feature_name]} <= {threshold:.2f}?\\n"
            f"훈련용 사람 수: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="#FFF0C2"];'
        )

        left_child = tree_structure.children_left[node_id]
        right_child = tree_structure.children_right[node_id]

        # 왼쪽 가지는 조건이 참인 경우
        dot_lines.append(
            f'{node_id} -> {left_child} [label="예"];'
        )

        # 오른쪽 가지는 조건이 거짓인 경우
        dot_lines.append(
            f'{node_id} -> {right_child} [label="아니요"];'
        )

        make_node(left_child)
        make_node(right_child)


make_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)


# =====================================================
# 22. 의사결정트리 그림
# =====================================================
st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# =====================================================
# 23. 의사결정트리 요약
# =====================================================
st.subheader("의사결정트리 요약")

st.write(
    f"**답을 내는 마디는 모두 {leaf_count}칸이며, "
    f"그중 {negative_leaf_count}칸이 '뇌졸중 아님'이라고 답합니다.**"
)

if asked_features:

    asked_names = [
        KOREAN_NAMES[feature]
        for feature in selected_features
        if feature in asked_features
    ]

    st.write(
        "**이 나무가 실제로 물은 속성:** "
        + ", ".join(asked_names)
    )

else:
    st.write("이 나무는 질문을 만들지 않고 바로 답합니다.")


# =====================================================
# 24. 데이터 분리 안내
# =====================================================
st.divider()

st.header("5. 학습과 테스트 데이터 확인")

split_table = pd.DataFrame({
    "데이터": [
        "훈련 데이터",
        "테스트 데이터",
        "전체 데이터"
    ],
    "사람 수": [
        len(train_df),
        len(test_df),
        len(data)
    ],
    "비율(%)": [
        round(len(train_df) / len(data) * 100, 2),
        round(len(test_df) / len(data) * 100, 2),
        100.00
    ]
})

st.dataframe(
    split_table,
    use_container_width=True,
    hide_index=True
)

st.caption("""
데이터는 id 순서로 정렬한 뒤 10명씩 묶었습니다.
각 묶음의 앞 3명은 테스트용, 나머지 7명은 훈련용입니다.

BMI 결측값 대체와 표준화는 훈련 데이터에서 계산한 값만 사용했습니다.
의사결정트리는 최대 깊이 3, 마지막 마디의 최소 인원 5명,
난수 고정값 42로 학습했습니다.
""")

st.caption("""
주의: 이 실습 모델은 교육용입니다.
예측 결과를 실제 의료 진단이나 치료 판단에 사용해서는 안 됩니다.
""")
