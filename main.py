```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 분석하고, 전체 기간과 최근 20년의 "
    "기온 변화 추세를 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

CUTOFF_YEAR = 2025
MIN_OBSERVATIONS = 300


# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")

    df = df[required].copy()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"])
    df = df[df["날짜"].dt.year <= CUTOFF_YEAR].copy()

    yearly = (
        df.assign(연도=df["날짜"].dt.year)
        .groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATIONS
    ].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 위해 유효한 연도가 최소 2개 필요합니다.")
    st.stop()


# -----------------------------
# 전체 기간 회귀 분석
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope_all, intercept_all = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

yearly["전체기간 회귀기온"] = slope_all * x + intercept_all

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

# 기울기를 100년당 변화량으로 환산
change_all = slope_all * 100


# -----------------------------
# 최근 20년 회귀 분석
# -----------------------------
# 가장 최근에 분석에 포함된 연도를 기준으로
# 20개 연도(시작 연도 포함)를 선택
recent_start_year = end_year - 19

recent = yearly[
    (yearly["연도"] >= recent_start_year)
    & (yearly["연도"] <= end_year)
].copy()

recent_n = len(recent)

if recent_n >= 2:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)

    slope_recent, intercept_recent = np.polyfit(xr, yr, 1)
    change_recent = slope_recent * 100
    recent_corr = np.corrcoef(xr, yr)[0, 1]

    recent["최근20년 회귀기온"] = (
        slope_recent * xr + intercept_recent
    )
else:
    slope_recent = None
    intercept_recent = None
    change_recent = None
    recent_corr = None


# -----------------------------
# 분석 기간 안내
# -----------------------------
st.subheader("📊 분석 데이터")

c1, c2, c3 = st.columns(3)
c1.metric("전체 분석 연도 수", f"{n_years}년")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")

st.caption(
    f"2025년까지의 자료 중 유효 평균기온 관측일이 "
    f"{MIN_OBSERVATIONS}일 이상인 연도만 사용합니다. "
    "최근 20년 비교는 분석에 포함된 가장 최근 연도를 기준으로 "
    f"{recent_start_year}~{end_year}년을 대상으로 합니다."
)


# -----------------------------
# 100년당 기온 변화 비교
# -----------------------------
st.divider()
st.subheader("🌡️ 100년당 기온 변화 비교")

st.write(
    "회귀 직선의 연간 기울기에 100을 곱해 "
    "100년 동안의 선형 변화량으로 환산했습니다."
)

left, right = st.columns(2)

with left:
    st.markdown("### 전체 분석 기간")
    st.markdown(
        f"""
        <div style="
            padding: 24px 12px;
            border-radius: 14px;
            background: rgba(128,128,128,0.10);
            text-align: center;
            min-height: 150px;
        ">
            <div style="font-size: 17px;">
                {start_year}~{end_year}년
            </div>
            <div style="font-size: 38px; font-weight: 700;
                        margin: 12px 0;">
                {change_all:+.2f} °C
            </div>
            <div style="font-size: 14px;">
                100년당 변화량
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"회귀 분석에 사용한 연도: {n_years}개")

with right:
    st.markdown("### 최근 20년")
    if recent_n >= 2:
        st.markdown(
            f"""
            <div style="
                padding: 24px 12px;
                border-radius: 14px;
                background: rgba(128,128,128,0.10);
                text-align: center;
                min-height: 150px;
            ">
                <div style="font-size: 17px;">
                    {recent_start_year}~{end_year}년
                </div>
                <div style="font-size: 38px; font-weight: 700;
                            margin: 12px 0;">
                    {change_recent:+.2f} °C
                </div>
                <div style="font-size: 14px;">
                    100년당 변화량
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"회귀 분석에 사용한 연도: {recent_n}개")
    else:
        st.warning(
            "최근 20년 구간에 회귀 분석이 가능한 연도가 "
            "2개 미만입니다."
        )

if recent_n >= 2:
    difference = change_recent - change_all

    st.write(
        f"**두 추세의 차이:** 최근 20년 기울기가 전체 기간보다 "
        f"100년당 {difference:+.2f} °C입니다."
    )

    if difference > 0:
        st.write(
            "최근 20년의 선형 추세가 전체 기간의 추세보다 "
            "상승 방향으로 더 가파릅니다."
        )
    elif difference < 0:
        st.write(
            "최근 20년의 선형 추세가 전체 기간의 추세보다 "
            "상승 속도가 낮거나 하락 방향으로 더 가파릅니다."
        )
    else:
        st.write("두 기간의 회귀 기울기가 같습니다.")


# -----------------------------
# 상관계수
# -----------------------------
st.divider()
st.subheader("🔎 연도와 평균기온의 상관관계")

if recent_n >= 2:
    corr1, corr2 = st.columns(2)
    corr1.metric("전체 기간 상관계수", f"{correlation:.4f}")
    corr2.metric("최근 20년 상관계수", f"{recent_corr:.4f}")
else:
    st.metric("전체 기간 상관계수", f"{correlation:.4f}")


# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.divider()
st.subheader("📈 연도별 평균기온과 추세선")

fig = go.Figure()

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
            "관측일: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["전체기간 회귀기온"],
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 기간 추정: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

if recent_n >= 2:
    fig.add_trace(
        go.Scatter(
            x=recent["연도"],
            y=recent["최근20년 회귀기온"],
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                "연도: %{x}년<br>"
                "최근 20년 추정: %{y:.2f} °C"
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
# 연도별 기온 예측
# -----------------------------
st.divider()
st.subheader("🔮 선택한 연도의 예상 평균기온")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
)

predicted_temp = slope_all * selected_year + intercept_all

st.caption(
    "예측값은 전체 분석 기간의 회귀 직선을 연장해 계산합니다."
)

st.markdown(
    f"""
    <div style="
        padding: 28px 20px;
        border-radius: 16px;
        background: rgba(128,128,128,0.10);
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

prediction_fig = go.Figure(fig)

prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측값",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate=(
            f"{selected_year}년<br>"
            "예상 평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

line_start = min(start_year, selected_year)
line_end = max(end_year, selected_year)

prediction_fig.add_trace(
    go.Scatter(
        x=[line_start, line_end],
        y=[
            slope_all * line_start + intercept_all,
            slope_all * line_end + intercept_all,
        ],
        mode="lines",
        name="전체 기간 회귀식 연장",
        line=dict(dash="dot", width=2),
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
# 회귀식 및 해석
# -----------------------------
st.divider()
st.subheader("🧮 전체 기간 회귀식")

st.latex(
    rf"\hat{{T}} = {slope_all:.5f}\times\text{{연도}} "
    rf"{intercept_all:+.2f}"
)

st.write(f"- 전체 기간 기울기: {slope_all:+.5f} °C/년")
st.write(f"- 전체 기간 100년당 변화량: {change_all:+.2f} °C")

if recent_n >= 2:
    st.write(f"- 최근 20년 기울기: {slope_recent:+.5f} °C/년")
    st.write(f"- 최근 20년 100년당 변화량: {change_recent:+.2f} °C")

st.warning(
    "주의: 100년당 변화량은 각 기간의 회귀 기울기를 100배 한 "
    "통계적 환산값입니다. 특히 최근 20년의 기울기를 100년으로 "
    "환산한 값은 그 추세가 100년 동안 지속된다는 뜻이 아닙니다. "
    "이 분석은 미래의 실제 기온을 확정적으로 예측하지 않습니다."
)


# -----------------------------
# 분석 데이터 확인
# -----------------------------
with st.expander("연도별 분석 데이터 보기"):
    display_data = yearly.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)
    display_data["전체기간 회귀기온"] = (
        display_data["전체기간 회귀기온"].round(2)
    )
    st.dataframe(display_data, use_container_width=True)
```
