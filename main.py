```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형 회귀로 선택한 연도의 예상 평균기온을 계산합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)

CUTOFF_YEAR = 2025
MIN_OBSERVATIONS = 300


# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필수 열이 없습니다: {', '.join(missing_columns)}"
        )

    df = df[required_columns].copy()

    # 날짜와 기온을 올바른 자료형으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"])

    # 수업 기준 기간: 2025년까지
    df = df[df["날짜"].dt.year <= CUTOFF_YEAR].copy()

    # 연도별 평균기온 및 유효 관측일 수 계산
    yearly = (
        df.assign(연도=df["날짜"].dt.year)
        .groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATIONS
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    if yearly.empty:
        raise ValueError(
            "분석 조건을 만족하는 연도가 없습니다."
        )

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.stop()


# -----------------------------
# 선형 회귀 계산
# -----------------------------
if len(yearly) < 2:
    st.error("회귀 분석을 위해서는 유효한 연도가 최소 2개 필요합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

# y = 기울기 * x + 절편
slope, intercept = np.polyfit(x, y, 1)

# 피어슨 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀 직선에 따른 연평균기온
yearly["회귀예측기온"] = slope * x + intercept

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)


# -----------------------------
# 분석에 사용한 데이터 정보
# -----------------------------
st.subheader("📊 분석 데이터")

col1, col2, col3 = st.columns(3)

col1.metric("회귀에 사용한 연도 수", f"{n_years:,}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")

st.caption(
    f"분석 조건: 2025년까지의 자료 중 "
    f"유효 평균기온 관측일이 {MIN_OBSERVATIONS}일 이상인 연도. "
    "관측일수는 평균기온이 유효한 날짜의 개수로 계산합니다."
)

# -----------------------------
# 상관계수 표시
# -----------------------------
st.subheader("🔎 연도와 평균기온의 상관관계")

corr_col1, corr_col2 = st.columns(2)

corr_col1.metric("피어슨 상관계수", f"{correlation:.4f}")
corr_col2.metric("연도당 회귀 기울기", f"{slope:.4f} °C")

if correlation > 0:
    st.write(
        "연도가 증가할수록 연평균기온이 높아지는 "
        "양의 상관관계가 관찰됩니다."
    )
elif correlation < 0:
    st.write(
        "연도가 증가할수록 연평균기온이 낮아지는 "
        "음의 상관관계가 관찰됩니다."
    )
else:
    st.write("연도와 연평균기온 사이의 선형 상관관계가 거의 없습니다.")

# -----------------------------
# Plotly 산점도 + 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "유효 관측일: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 추정기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 연도 선택 및 예상 기온
# -----------------------------
st.divider()
st.subheader("🌡️ 미래 연도의 예상 평균기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
)

predicted_temp = slope * selected_year + intercept

st.caption(
    f"{start_year}~{end_year}년의 연평균기온 자료로 만든 "
    "선형 회귀식을 이용한 추정값입니다."
)

st.markdown(
    f"""
    <div style="
        padding: 28px 20px;
        border-radius: 16px;
        background: rgba(128, 128, 128, 0.10);
        text-align: center;
        margin: 12px 0 24px 0;
    ">
        <div style="font-size: 22px; margin-bottom: 8px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size: 54px; font-weight: 700;">
            {predicted_temp:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 선택 연도를 그래프에 표시
# -----------------------------
prediction_fig = go.Figure(fig)

prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 추정값",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate=(
            f"{selected_year}년<br>"
            "예상 평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 예측 연도가 과거 자료 범위 밖이어도 직선이 이어지도록 표시
line_start = min(start_year, selected_year)
line_end = max(end_year, selected_year)

prediction_fig.add_trace(
    go.Scatter(
        x=[line_start, line_end],
        y=[
            slope * line_start + intercept,
            slope * line_end + intercept,
        ],
        mode="lines",
        name="선택 연도까지의 회귀 추세",
        line=dict(dash="dash", width=2),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 추정기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
)

st.plotly_chart(
    prediction_fig,
    use_container_width=True,
    key="prediction_chart",
)

# -----------------------------
# 회귀식과 해석
# -----------------------------
st.divider()
st.subheader("🧮 회귀식")

st.latex(
    rf"\hat{{T}} = {slope:.5f}\times \text{{연도}} "
    rf"{intercept:+.2f}"
)

st.write(
    f"- 회귀 분석에 사용한 연도: {n_years}개"
)
st.write(
    f"- 분석 시작 연도: {start_year}년"
)
st.write(
    f"- 분석 끝 연도: {end_year}년"
)
st.write(
    f"- 연도 1년 증가에 따른 회귀 추정 변화: "
    f"{slope:+.4f} °C"
)
st.write(
    f"- 피어슨 상관계수: {correlation:.4f}"
)

st.warning(
    "주의: 이 결과는 과거 연평균기온과 연도 사이의 "
    "선형 관계를 연장한 통계적 추정입니다. "
    "온실가스 배출, 자연 변동, 도시 환경 변화 등 "
    "기후 요인을 별도로 반영하지 않으므로 "
    "실제 미래 기온 예보로 해석해서는 안 됩니다."
)

# -----------------------------
# 분석에 사용된 연도별 데이터
# -----------------------------
with st.expander("연도별 분석 데이터 보기"):
    display_data = yearly.copy()
    display_data["연도"] = display_data["연도"].astype(int)
    display_data["연평균기온"] = display_data["연평균기온"].round(2)
    display_data["회귀예측기온"] = display_data["회귀예측기온"].round(2)
    st.dataframe(display_data, use_container_width=True)
```
