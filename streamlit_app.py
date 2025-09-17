# streamlit_app.py
"""
Streamlit 앱: 실외 공개 데이터 대시보드 + 사용자 입력(보고서 기반) 대시보드
- 공개 데이터: Our World in Data PM2.5 CSV 사용(대체/재시도 로직 포함)
  출처: https://ourworldindata.org/grapher/average-exposure-pm25-pollution.csv
- 사용자 입력: 사용자가 제공한 보고서 텍스트를 바탕으로 생성한 예시/요약 데이터프레임 사용
구현 규칙 요약:
- 모든 UI 한국어
- 전처리: date,value,group(optional)
- 미래 데이터 제거(지역 로컬타임 Asia/Seoul 기준)
- @st.cache_data 사용
- 전처리된 데이터 CSV 다운로드 버튼 제공
- 폰트 시도: /fonts/Pretendard-Bold.ttf (없으면 무시)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import pycountry
from datetime import datetime
import pytz
import time
import plotly.express as px

# ---------------------------
# 설정
# ---------------------------
st.set_page_config(page_title="실내·실외 공기질 대시보드", layout="wide")
LOCAL_TZ = "Asia/Seoul"

# Pretendard 폰트 시도 (없으면 자동 생략)
st.markdown(
    """
    <style>
    @font-face {
        font-family: 'PretendardLocal';
        src: url('/fonts/Pretendard-Bold.ttf') format('truetype');
        font-weight: 700;
    }
    html, body, [class*="css"]  {
        font-family: PretendardLocal, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# 유틸리티
# ---------------------------
def now_seoul():
    return pd.Timestamp.now(tz=pytz.timezone(LOCAL_TZ))

def remove_future_dates(df, date_col="date"):
    """date_col can be datetime or year numeric. Remove rows after local midnight today."""
    try:
        today = now_seoul().normalize()
        if pd.api.types.is_integer_dtype(df[date_col]) or pd.api.types.is_float_dtype(df[date_col]):
            # treat as year
            df = df[df[date_col] <= int(today.year)]
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[df[date_col].notna()]
            df = df[df[date_col] <= today]
    except Exception:
        pass
    return df

# ---------------------------
# 공개 데이터: Our World in Data PM2.5
# 출처 주석: https://ourworldindata.org/grapher/average-exposure-pm25-pollution.csv
# ---------------------------
DATA_URL_OWID = "https://ourworldindata.org/grapher/average-exposure-pm25-pollution.csv?v=1&csvType=full&useColumnShortNames=true"

@st.cache_data(ttl=3600)
def fetch_owid_pm25(url=DATA_URL_OWID, max_retries=2, timeout=10):
    last_exc = None
    for attempt in range(max_retries+1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            return df
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt*2)
    # 실패 시 None
    return None

@st.cache_data
def prepare_owid_df(raw_df):
    df = raw_df.copy()
    # 컬럼 자동 탐지: PM2.5 관련 값 칼럼 찾기
    def find_value_column(cols):
        for c in cols:
            low = c.lower()
            if "pm2" in low or "pm 2" in low or "pm25" in low or "average exposure" in low:
                return c
        for c in cols:
            if c.lower() in ("value","avg","mean"):
                return c
        return None

    value_col = find_value_column(df.columns)
    if value_col is None:
        raise RuntimeError("PM2.5 값 칼럼을 찾을 수 없습니다")

    # 표준화: country, iso_alpha(Code), year, value
    if "Entity" in df.columns:
        df["country"] = df["Entity"]
    elif "entity" in df.columns:
        df["country"] = df["entity"]
    else:
        df["country"] = df.iloc[:,0].astype(str)

    if "Year" in df.columns:
        df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        df["year"] = pd.to_numeric(df.iloc[:,2], errors="coerce").astype("Int64")

    if "Code" in df.columns:
        df["iso_alpha"] = df["Code"]
    else:
        df["iso_alpha"] = None

    df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["value","country","year"])
    # iso가 없는 경우 pycountry로 시도
    mask = df["iso_alpha"].isna() | (df["iso_alpha"] == "")
    if mask.any():
        def name_to_iso(name):
            try:
                c = pycountry.countries.lookup(name)
                return c.alpha_3
            except Exception:
                return None
        df.loc[mask, "iso_alpha"] = df.loc[mask, "country"].apply(name_to_iso)
    df = df.dropna(subset=["iso_alpha"])
    df["iso_alpha"] = df["iso_alpha"].astype(str)
    df = df[["country","iso_alpha","year","value"]].rename(columns={"year":"year","value":"value"})
    return df

# ---------------------------
# 사용자 입력(보고서 기반) 데이터 생성
# - 입력으로 주어진 보고서 텍스트만 사용
# - 표준화: date,value,group(optional)
# ---------------------------
@st.cache_data
def build_user_datasets():
    # 1) 생활패턴: 하루 실내체류 비율(단일 값) -> 보고서 내용 반영
    df_time = pd.DataFrame({
        "date": [pd.Timestamp(f"{year}-01-01") for year in range(2000, 2024)],
        "value": [95.0 + np.random.normal(0, 0.5) for _ in range(2000, 2024)], # 95% 이상으로 설정
        "group": ["실내 체류 비율(%)"]*len(range(2000, 2024))
    })

    # 2) WHO 추산: 대기 오염으로 인한 사망자 중 실내 공기 오염 관련 비율 (정적)
    who_mortality_df = pd.DataFrame({
        "group": ["실내 공기 오염 관련 사망자 비율"],
        "value": [93.0],
        "date": pd.to_datetime(["2020-01-01"]) # 가상 날짜
    })

    # 3) 실내 공기질 관리 사각지대 (측정/점검 비율)
    management_gap_df = pd.DataFrame({
        "group": ["실내 공기질 측정 및 점검 비율"],
        "value": [20.0], # 20% 미만으로 표현
        "date": pd.to_datetime(["2022-01-01"])
    })

    # 4) 예방 방법 선호도 (보고서 내용 기반으로 재구성)
    prevention_methods = {
        "학교: 공기청정기 설치 및 환기 점검": 30,
        "가정: 규칙적 환기 및 실내 흡연 금지": 40,
        "국가: 실내공기질 관리법 강화": 20,
        "기타/학생 실천": 10
    }
    prevention_df = pd.DataFrame({
        "group": list(prevention_methods.keys()),
        "value": list(prevention_methods.values()),
        "date": pd.to_datetime(["2023-01-01"]*len(prevention_methods))
    })

    # 5) 민감시설별 예시 측정값 (기존 유지, 보고서 맥락에 맞춰 설명)
    facilities = ["산후조리원","어린이집","지하역사","학원","오래된 교실"] # 오래된 교실 추가
    # 예시: PM2.5 (µg/m3), CO2 (ppm), 포름알데히드 (µg/m3), 세균 CFU/m3
    rows = []
    rng = np.random.RandomState(42)
    for year in range(2007,2018):
        for f in facilities:
            rows.append({
                "date": pd.Timestamp(f"{year}-06-30"),
                "group": f,
                "PM2.5": max(5, float(rng.normal(20 + (0 if f not in ["지하역사", "오래된 교실"] else 10), 5))),
                "CO2": max(400, float(rng.normal(800 + (200 if f in ['지하역사','학원','오래된 교실'] else 0), 120))),
                "폼알데히드": max(10, float(rng.normal(30 + (20 if f=="산후조리원" else 0), 8))),
                "세균": max(50, float(rng.normal(300 + (150 if f=="어린이집" else 0), 80)))
            })
    fac_df = pd.DataFrame(rows)
    # melt 표준화: date,value,group(pollutant/facility)
    fac_long = fac_df.melt(id_vars=["date","group"], var_name="pollutant", value_name="value")
    # group 컬럼을 "시설"로 남기고 pollutant 별로 구분
    fac_long["group_full"] = fac_long["group"] + " | " + fac_long["pollutant"]
    fac_long = fac_long.rename(columns={"group":"facility"})
    fac_long = fac_long[["date","value","facility","pollutant","group_full"]]

    return {
        "time_df": df_time,
        "who_mortality_df": who_mortality_df,
        "management_gap_df": management_gap_df,
        "prevention_df": prevention_df,
        "facility_long_df": fac_long
    }

# ---------------------------
# 앱 레이아웃: 탭
# ---------------------------
st.title("실내 공기질, 실외 공기질 얼마나 차이날까? (가제)")
st.markdown("### 서론 (문제 제기)")
st.markdown("우리는 하루 대부분을 학교와 집 같은 실내에서 생활한다. 뉴스나 SNS에서 실외 미세먼지나 황사에 대한 경고는 쉽게 볼 수 있지만 정작 우리가 가장 오랜 시간을 보내는 실내 공기질이 얼마나 나쁜지, 그리고 그것이 우리 건강과 학습에 어떤 영향을 미치는지는 잘 알려지지 않는다. 그래서 우리는 실내 공기질과 실외 공기질의 차이를 직접 비교하고, 청소년으로서 우리가 할 수 있는 대응책을 찾아보기 위해 이 보고서를 작성했다.")
st.markdown("---")


tabs = st.tabs(["본론 1: 데이터 분석(전세계 PM2.5)", "본론 2: 원인 및 영향 탐구(보고서 요약)", "결론: 제언"])

# ---------- 탭1: 공개 데이터 ----------
with tabs[0]:
    st.header("본론 1: 데이터 분석 - 전 세계 PM2.5 노출")
    st.caption("데이터 출처: Our World in Data CSV. 실패 시 예시 데이터로 대체됩니다.")
    
    st.markdown("전 세계 대기질 지수(AQI) 자료를 보면, 가장 오염도가 높은 나라는 인도와 중국이며, 한국은 상위 20위권 내에서 꾸준히 높은 수치를 기록하고 있다. 반대로 가장 청정한 나라는 핀란드, 아이슬란드와 같은 북유럽 국가다. 오염도가 높은 국가의 청소년은 학습 환경에서 집중력이 떨어지고, 호흡기 질환과 알레르기 같은 건강 문제를 겪을 가능성이 높다.")
    st.markdown("국제 사례를 보면, 공기질 관리가 잘 된 나라에서는 청소년 건강과 학습권 보호를 위한 구체적인 정책을 시행하고 있다.")
    st.markdown("- **핀란드**: 학교와 공공시설에 공기질 모니터링 의무화")
    st.markdown("- **호주**: 산불 등 대기질 악화 시, 실내 대피 지침 마련")
    st.markdown("- **WHO**: 환기·필터 개선, 실내 금연, 조기 경보 체계 구축 권고")
    st.markdown("이처럼 실내 공기질을 관리하는 체계가 마련되어 있어야 청소년들이 안전하게 학습할 수 있다.")
    
    st.markdown("---")
    st.subheader("전 세계 PM2.5 노출 현황")

    raw = fetch_owid_pm25()
    if raw is None:
        st.error("공개 데이터 불러오기 실패. 예시 데이터로 자동 대체합니다.")
        # 예시 대체 데이터 생성(소수 국가, 연도)
        sample = pd.DataFrame({
            "country":["South Korea","China","India","Finland","Iceland"],
            "iso_alpha":["KOR","CHN","IND","FIN","ISL"],
            "year":[2015,2015,2015,2015,2015],
            "value":[25.0,85.0,95.0,6.0,5.0]
        })
        df_pm = sample
        st.info("네트워크 오류로 인해 예시 샘플 데이터가 사용되었습니다. 실제 데이터가 필요하면 인터넷 연결을 확인하세요.")
    else:
        try:
            df_pm = prepare_owid_df(raw)
        except Exception as e:
            st.error("데이터 전처리 중 오류 발생 예시 데이터 사용")
            df_pm = pd.DataFrame({
                "country":["South Korea","China","India","Finland","Iceland"],
                "iso_alpha":["KOR","CHN","IND","FIN","ISL"],
                "year":[2015,2015,2015,2015,2015],
                "value":[25.0,85.0,95.0,6.0,5.0]
            })

    # 미래 데이터 제거(연도 기반)
    df_pm = remove_future_dates(df_pm, date_col="year")

    # 사이드바 설정(탭 내)
    st.sidebar.header("공개 데이터 설정")
    years = sorted(df_pm["year"].unique())
    if len(years) == 0:
        st.warning("표시할 연도 데이터가 없습니다.")
    else:
        year_min, year_max = int(min(years)), int(max(years))
        year_choice = st.sidebar.slider("연도 선택", year_min, year_max, year_max)
        animate = st.sidebar.checkbox("연도 애니메이션(가능한 경우)", value=True)
        vmin = st.sidebar.number_input("컬러 최소값(µg/m³)", value=0.0, format="%.1f")
        vmax = st.sidebar.number_input("컬러 최대값(µg/m³)", value=60.0, format="%.1f")
        # 다운로드
        st.sidebar.download_button("처리된 공개 데이터 다운로드 (CSV)", data=df_pm.to_csv(index=False).encode("utf-8"), file_name="owid_pm25_processed.csv", mime="text/csv")

        # 시각화
        if animate and df_pm["year"].nunique() > 1:
            fig = px.choropleth(
                df_pm,
                locations="iso_alpha",
                color="value",
                hover_name="country",
                animation_frame="year",
                range_color=(vmin, vmax),
                labels={"value":"PM2.5 µg/m³"},
                projection="natural earth"
            )
            fig.update_layout(coloraxis_colorbar=dict(title="PM2.5 µg/m³"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            df_sel = df_pm[df_pm["year"] == int(year_choice)]
            if df_sel.empty:
                st.warning("선택한 연도에 데이터가 없습니다.")
            else:
                fig = px.choropleth(
                    df_sel,
                    locations="iso_alpha",
                    color="value",
                    hover_name="country",
                    range_color=(vmin, vmax),
                    labels={"value":"PM2.5 µg/m³"},
                    projection="natural earth"
                )
                fig.update_layout(title_text=f"PM2.5 평균 노출 {year_choice}", coloraxis_colorbar=dict(title="PM2.5 µg/m³"))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 메모")
    st.markdown("- 일부 집계(예: World)는 ISO 코드가 없어 제외됩니다.")
    st.markdown("- pycountry가 설치되어 있으면 이름→ISO 변환을 시도합니다.")
    st.markdown("- 소스: Our World in Data CSV. 코드 주석에 출처 URL 포함.")

# ---------- 탭2: 사용자 입력 기반 대시보드 ----------
with tabs[1]:
    st.header("본론 2: 원인 및 영향 탐구 (보고서 요약 기반)")
    st.caption("입력: 사용자가 제공한 보고서 텍스트를 바탕으로 생성한 요약/예시 데이터만 사용합니다. 앱 실행 중 별도 업로드 불필요")

    datasets = build_user_datasets()

    # 요약 카드
    col1, col2, col3 = st.columns(3)
    col1.metric("평균 실내 체류 비율", "95% 이상")
    col2.metric("WHO 추산: 실내 공기 오염 관련 사망", "93%")
    col3.metric("실내 공기질 관리 사각지대", "측정/점검 20% 미만")

    st.markdown("---")
    st.subheader("1. 한국인의 실내 체류 시간(연도별)")
    st.markdown("WHO 추산에 따르면, 대기 오염으로 인한 사망자 중 약 93%가 실내 공기 오염과 관련이 있다. 우리 일상에서 실내 체류 시간은 하루 평균 95% 이상으로, 학교·학원·어린이집 등 청소년이 자주 머무는 공간의 공기질 관리가 무엇보다 중요하다.")
    time_df = datasets["time_df"].copy()
    time_df = remove_future_dates(time_df, date_col="date")
    # 시계열 그래프
    fig_time = px.line(time_df, x="date", y="value", title="한국인 하루 실내 체류 비율 추이", labels={"value":"비율(%)", "date":"연도"})
    st.plotly_chart(fig_time, use_container_width=True)

    st.download_button("실내 체류 데이터 다운로드 (CSV)", data=time_df.to_csv(index=False).encode("utf-8"), file_name="user_time_data.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("2. 실내 공기질 관리 현황 및 예방 방법")
    st.markdown("그러나 실제로는 실내 공기질 측정과 점검이 20% 미만으로, 관리 사각지대가 존재한다. 특히 오래된 건물일수록, 환기 설비가 부족한 공간일수록 오염물질 농도가 높게 나타난다.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 실내 공기질 측정 및 점검 현황")
        management_gap_df = datasets["management_gap_df"]
        fig1 = px.bar(management_gap_df, x="group", y="value", labels={"group":"항목","value":"비율(%)"}, title="실내 공기질 측정/점검 비율 (예시)")
        st.plotly_chart(fig1, use_container_width=True)
        st.download_button("관리 사각지대 데이터 다운로드 (CSV)", data=management_gap_df.to_csv(index=False).encode("utf-8"), file_name="management_gap.csv", mime="text/csv")
    with c2:
        st.markdown("#### 예방 방법 선호도 (보고서 기반)")
        prevention_df = datasets["prevention_df"]
        fig2 = px.pie(prevention_df, names="group", values="value", title="실내 공기질 예방 방법 (보고서 기반)")
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button("예방 방법 데이터 다운로드 (CSV)", data=prevention_df.to_csv(index=False).encode("utf-8"), file_name="prevention_methods.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("3. 민감시설 및 오래된 공간의 공기질 예시 측정값")
    st.markdown("민감시설(산후조리원, 어린이집 등)과 오래된 교실은 특정 오염물질이 더 높게 관측되는 경향이 있습니다. 이는 실내 공기질 관리의 중요성을 보여줍니다.")
    fac_long = datasets["facility_long_df"].copy()
    fac_long = remove_future_dates(fac_long, date_col="date")

    # 사이드바 컨트롤(탭 내부)
    st.sidebar.header("사용자 데이터 필터")
    pollutant_options = fac_long["pollutant"].unique().tolist()
    selected_pollutant = st.sidebar.selectbox("오염물질 선택", pollutant_options, index=pollutant_options.index("PM2.5") if "PM2.5" in pollutant_options else 0)
    facilities = fac_long["facility"].unique().tolist()
    selected_facilities = st.sidebar.multiselect("시설 선택(여러개 선택 가능)", facilities, default=facilities[:3])
    smooth = st.sidebar.checkbox("이동평균(3포인트) 적용", value=False)
    min_year = int(fac_long["date"].dt.year.min())
    max_year = int(fac_long["date"].dt.year.max())
    year_range = st.sidebar.slider("연도 범위", min_year, max_year, (min_year, max_year))

    # 필터 적용
    df_plot = fac_long[fac_long["pollutant"] == selected_pollutant]
    df_plot = df_plot[df_plot["facility"].isin(selected_facilities)]
    df_plot = df_plot[(df_plot["date"].dt.year >= year_range[0]) & (df_plot["date"].dt.year <= year_range[1])]

    if df_plot.empty:
        st.warning("선택 조건에 맞는 데이터가 없습니다. 필터를 조정하세요.")
    else:
        # pivot for plotting
        pivot = df_plot.pivot_table(index="date", columns="facility", values="value")
        if smooth:
            pivot = pivot.rolling(3, min_periods=1).mean()
        fig_fac = px.line(pivot.reset_index(), x="date", y=pivot.columns, labels={"value":"값","date":"날짜"}, title=f"{selected_pollutant} 측정값 추이 (민감시설 예시)")
        st.plotly_chart(fig_fac, use_container_width=True)
        st.download_button("민감시설 측정값 CSV 다운로드", data=df_plot.to_csv(index=False).encode("utf-8"), file_name="facility_pollutant_timeseries.csv", mime="text/csv")

# ---------- 탭3: 결론 및 제언 ----------
with tabs[2]:
    st.header("결론 (제언): 그래서 우리는 무엇을 해야 할까?")
    st.markdown("이번 보고서를 통해 우리는 데이터를 직접 확인하며, 실외 공기뿐 아니라 실내 공기질이 청소년 건강과 학습환경에 큰 영향을 준다는 사실을 알게 되었다. 우리가 매일 보내는 학교와 학원, 집이 단순한 생활 공간이 아니라, 호흡과 집중력에 직접적으로 연결된 공간임을 깨달은 것이다.")
    st.markdown("이제 문제를 아는 것을 넘어, 실질적인 해결을 위해 행동할 때다. 단순히 어른들이 만들어 줄 변화를 기다리기보다, 청소년 스스로 작은 실천을 시작하는 것이 중요하다. 우리의 노력은 공기청정기나 환기 설비 같은 물리적 장치와 결합될 때 비로소 실질적 변화를 만들어낼 수 있다.")
    st.markdown("따라서 우리는 다음과 같은 세 가지 행동을 제안한다.")

    st.markdown("---")
    st.subheader("1. 제언 1: ‘공기질 데이터 탐사대’ – 정확히 알고, 친구들에게 알리기")
    st.markdown("실내 공기질 문제는 막연한 불안감으로는 해결되지 않는다. 정확한 데이터를 통해 문제를 이해하고, 이를 친구들과 공유하는 것이 행동의 첫걸음이다.")
    st.markdown("- 우리 학교에 ‘실내 공기질 분석반’ 같은 동아리를 만들어, WHO, 한국환경공단, 국내외 공기질 측정 사이트에서 데이터를 직접 내려받아 분석한다. 예를 들어 교실별 PM2.5 수치, 이산화탄소 농도 변화, 환기 시간대별 공기질 변화 등을 비교할 수 있다.")
    st.markdown("- 분석 결과는 카드뉴스, 짧은 영상, 포스터 형태로 제작해 학교 SNS, 급식실, 복도 게시판 등에 배포한다. 예를 들어 ‘최근 5년간 우리 학교 교실 평균 PM2.5 변화’ 같은 제목은 학생들의 관심을 쉽게 끌 수 있다.")
    st.markdown("- 또한, 통계 수행평가나 과학 탐구 보고서, 사회문화 시간의 환경 문제 프로젝트 등과 연계해, 학교 수업과 생활 속 탐구를 연결하면 학습 효과도 높일 수 있다. 이렇게 데이터 기반으로 문제를 인식하고 공유하면, 실질적인 변화를 위한 근거가 마련된다.")
    st.image("https://images.unsplash.com/photo-1579547622329-87309990ee42?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80", caption="데이터 분석을 통해 공기질 문제를 이해하고 알리기")


    st.markdown("---")
    st.subheader("2. 제언 2: ‘우리 교실, 1단계 공기질 지키기’ – 실천 가능한 생활 속 행동")
    st.markdown("실내 공기질 개선은 장기적인 설비 투자뿐 아니라, 지금 당장 학생들이 실천할 수 있는 작은 습관에서 시작할 수 있다.")
    st.markdown("- **환기 습관화**: 점심시간, 이동수업, 쉬는 시간마다 ‘칼환기’를 실시한다. 교실 앞뒤 창문을 2~3분만 열어도 공기 순환이 이루어지고, 쌓인 이산화탄소와 미세먼지가 빠져나간다.")
    st.markdown("- **불필요한 전자기기 끄기**: 교실이나 학원에서 사용하지 않는 컴퓨터, 프로젝터, 선풍기, 조명을 끄는 습관을 정착시킨다. 실내 온도와 공기 순환에도 긍정적 영향을 준다.")
    st.markdown("- **공기청정기·식물 활용**: 학급별 공기청정기를 활용하고, 가능하다면 공기 정화 기능이 있는 식물을 교실에 배치한다. 이는 미세먼지 저감과 함께 시각적 안정감도 준다.")
    st.markdown("- **햇빛·온도 관리**: 햇빛이 강한 오후 1~3시에는 블라인드를 내려 실내 온도 상승과 공기질 악화를 방지한다.")
    st.markdown("작은 습관이 쌓이면, 교실 환경은 크게 달라질 수 있으며, 학생 스스로도 자신의 건강과 학습 환경을 지킬 수 있다.")
    st.image("https://images.unsplash.com/photo-1587620931557-05c742e88a3b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80", caption="교실에서 실천할 수 있는 공기질 개선 습관")


    st.markdown("---")
    st.subheader("3. 제언 3: ‘데이터로 말하기’ – 어른들에게 우리의 목소리 전달하기")
    st.markdown("학생의 노력만으로는 학교 전체 공기질을 바꾸기 어렵다. 따라서 데이터 기반으로 논리적인 요구를 만들어, 학교와 교육청의 지원을 이끌어내야 한다.")
    st.markdown("- **교실별 공기질 기록**: 시간대별 공기질 측정, 낡은 환기 설비·에어컨 실태 조사, 필터 교체 주기 기록 등 구체적인 데이터를 남긴다. 사진과 표를 함께 기록하면 더 강력한 증거가 된다.")
    st.markdown("- **학생회 활용**: 측정 자료를 바탕으로 ‘안전하고 깨끗한 교실을 만들어주세요!’라는 안건을 학생회에 제출하고, 전교생 서명 운동으로 의견을 모은다.")
    st.markdown("- **정책 제안**: 학급 대표로 학생회 회의에 참여하거나, 교육청 ‘국민신문고’, 시청 ‘주민참여예산’ 제도를 활용해 구체적 정책을 제안한다. 예를 들어 “교실 공기청정기 필터 교체 주기를 단축해주세요” 또는 “옥상에 녹색 공간과 차열 페인트 설치” 같은 실현 가능한 안을 요구할 수 있다.")
    st.markdown("데이터와 논리적인 요구가 모이면, 단순히 호소하는 것보다 훨씬 강력하게 정책 변화를 유도할 수 있다. 우리의 작은 실천과 꾸준한 목소리가 합쳐질 때, 비로소 학교를 청정하고 안전한 학습 공간으로 바꿀 수 있다.")
    st.image("https://images.unsplash.com/photo-1543269865-cbf427fdc1ae?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80", caption="데이터 기반으로 정책 변화를 제안하는 학생들")


    st.info("주의: 보고서 기반 대시보드의 수치는 제공된 보고")