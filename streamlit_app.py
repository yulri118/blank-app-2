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

변경사항(피드백 반영):
- 대시보드에 사용된 모든 외부 이미지 제거
- 제목에서 '(가제)' 제거 및 문제제기 문단 교체(피드백6)
- 탭을 5개(요구에 맞춘 5개 선택창)로 재구성하고 텍스트 통일
- 실내/실외 비교 그래프 추가
- 실내 측정/점검 현황 데이터는 연도별 바 차트로 수정(단일 20%만 있는 문제 해결)
- 대기질 개선 유도 기능 추가: 건강·학습 효과 계산기, 오늘의 교실 공기질 알리미, 교실 식물 효과 계산기, 폐 건강 위험 예측기, 예방 체크리스트(진행도)

"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import pycountry
import pytz
import time
import plotly.express as px
from datetime import datetime

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
        if date_col not in df.columns:
            return df
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
# 공개 데이터: 로컬 CSV 사용
# ---------------------------
DATA_PATH_OWID = "average-exposure-pm25-pollution.csv"  # 로컬 파일

@st.cache_data
def fetch_owid_pm25_local(path=DATA_PATH_OWID):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
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

    # 3) 실내 공기질 관리 사각지대 (측정/점검 비율) - 연도별 데이터로 확장
    years = list(range(2018, 2024))
    perc = [40, 35, 30, 25, 22, 20]  # 예시 추세: 점검 비율 감소 추세
    management_gap_df = pd.DataFrame({
        "date": pd.to_datetime([f"{y}-01-01" for y in years]),
        "value": perc,
        "group": ["실내 공기질 측정 및 점검 비율"]*len(years)
    })

    # 4) 예방 방법 선호도 (보고서 내용 기반으로 재구성)
    prevention_methods = {
        "학교: 공기청정기 설치 및 환기 점검": 30,
        "가정: 규칙적 환기 및 실내 흡연 금지": 40,
        "국가: 실내공기질 관리법 강화": 20,
        "학생 실천": 10
    }
    prevention_df = pd.DataFrame({
        "group": list(prevention_methods.keys()),
        "value": list(prevention_methods.values()),
        "date": pd.to_datetime(["2023-01-01"]*len(prevention_methods))
    })

    # 5) 민감시설별 예시 측정값 (기존 유지, 보고서 맥락에 맞춰 설명)
    facilities = ["산후조리원","어린이집","지하역사","학원","오래된 교실"]
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
    fac_long = fac_df.melt(id_vars=["date","group"], var_name="pollutant", value_name="value")
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
# 탭 구성(요구: 5개 선택창으로 통일)
# 순서 요구사항 반영: 전세계 PM2.5 전개 -> 실내 공기질 측정값(비교 포함) -> 보고서 페이지 -> 예방 방법(마지막)
# ---------------------------
TABS = [
    "데이터 분석: 전세계 PM2.5",
    "실내·실외 비교(측정값)",
    "보고서: 종합 분석",
    "예방 방법 및 계산기",
    "제언 및 행동"
]

# 상단 문제 제기(피드백6 텍스트로 교체)
st.markdown("# 실내 공기질과 실외 공기질: 청소년 건강을 위한 데이터 비교")
st.markdown(
    "현대 사회에서 사람들은 생활 시간의 대부분을 실내 공간에서 보낸다. 그러나 대기 오염에 대한 논의는 주로 실외 환경, 즉 미세먼지나 황사와 같은 외부 요인에 집중되어 있다. 이에 비해 실내 공기질은 상대적으로 관심을 덜 받아 왔으며, 그 위험성과 건강에 미치는 영향 또한 충분히 다뤄지지 않았다. 특히 청소년은 학교와 가정 등 제한된 공간에서 장시간 생활하기 때문에 실내 공기질의 영향을 직접적으로 받을 수밖에 없다. 본 보고서는 실내와 실외 공기질의 차이를 데이터로 비교·분석하고, 청소년의 건강 및 학습 환경에 미치는 영향을 검토하며, 이를 개선하기 위한 대응 방안을 제안하고자 한다."
)
st.markdown("---")

tabs = st.tabs(TABS)

# ---------- 탭0: 전세계 PM2.5 (지도) ----------
with tabs[0]:
    st.header("전세계 PM2.5 노출 현황 (지도)")
    st.caption("데이터 출처: Our World in Data CSV. 실패 시 예시 데이터로 대체됩니다.")

    raw = fetch_owid_pm25_local()
    if raw is None:
        st.error("공개 데이터 불러오기 실패. 예시 데이터로 자동 대체합니다.")
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
        except Exception:
            st.error("데이터 전처리 중 오류 발생. 예시 데이터 사용")
            df_pm = pd.DataFrame({
                "country":["South Korea","China","India","Finland","Iceland"],
                "iso_alpha":["KOR","CHN","IND","FIN","ISL"],
                "year":[2015,2015,2015,2015,2015],
                "value":[25.0,85.0,95.0,6.0,5.0]
            })

    df_pm = remove_future_dates(df_pm, date_col="year")

    st.sidebar.header("공개 데이터 설정")
    years = sorted(df_pm["year"].unique()) if "year" in df_pm.columns else []
    if len(years) == 0:
        st.warning("표시할 연도 데이터가 없습니다.")
        year_choice = None # year_choice가 None이 될 수 있도록 초기화
    elif len(years) == 1:
        # 연도가 하나뿐일 경우 슬라이더 대신 해당 연도를 표시
        year_choice = years[0]
        st.sidebar.write(f"선택 가능 연도: **{year_choice}** (데이터가 하나의 연도만 포함합니다.)")
        animate = False # 애니메이션 비활성화
        st.sidebar.checkbox("연도 애니메이션(가능한 경우)", value=False, disabled=True) # 체크박스 비활성화
    else:
        # 연도가 두 개 이상일 경우 정상적으로 슬라이더 표시
        year_min, year_max = int(min(years)), int(max(years))
        year_choice = st.sidebar.slider("연도 선택", year_min, year_max, year_max)
        animate = st.sidebar.checkbox("연도 애니메이션(가능한 경우)", value=True)

    # 이 아래는 year_choice가 유효할 때만 실행
    if year_choice is not None:
        vmin = st.sidebar.number_input("컬러 최소값(µg/m³)", value=0.0, format="%.1f")
        vmax = st.sidebar.number_input("컬러 최대값(µg/m³)", value=60.0, format="%.1f")
        st.sidebar.download_button("처리된 공개 데이터 다운로드 (CSV)", data=df_pm.to_csv(index=False).encode("utf-8"), file_name="owid_pm25_processed.csv", mime="text/csv")

        if animate and df_pm["year"].nunique() > 1: # 애니메이션 조건도 2개 이상의 연도일 때만 작동하도록 수정
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

# ---------- 탭1: 실내·실외 비교(측정값) ----------
with tabs[1]:
    st.header("실내·실외 공기질 비교")
    st.caption("실내 측정값 예시(보고서 기반)와 선택한 국가/연도의 실외 PM2.5를 함께 비교합니다.")

    datasets = build_user_datasets()
    fac_long = datasets["facility_long_df"].copy()
    fac_long = remove_future_dates(fac_long, date_col="date")

    # 사이드바(이 탭 전용)
    st.sidebar.header("실내·실외 비교 설정")
    # 선택: 비교할 국가(외부 데이터에서 추출) 및 연도
    countries = df_pm["country"].unique().tolist() if 'country' in df_pm.columns else ["South Korea"]
    country_choice = st.sidebar.selectbox("외부(국가) 선택", countries, index=countries.index("South Korea") if "South Korea" in countries else 0)
    year_choice_comp = st.sidebar.selectbox("비교 연도 선택", sorted(df_pm["year"].unique())[::-1] if 'year' in df_pm.columns else [2015])

    # 실외 PM2.5 값 선택 (데이터가 있으면 사용, 없으면 사용자 입력)
    outdoor_val = None
    df_pm_sel = df_pm[(df_pm["country"] == country_choice) & (df_pm["year"] == int(year_choice_comp))]
    if not df_pm_sel.empty:
        outdoor_val = float(df_pm_sel["value"].mean())
    else:
        outdoor_val = st.sidebar.number_input("외부 PM2.5 (µg/m³) 직접 입력", value=25.0)

    # 실내 평균 PM2.5 계산(시설별로 평균)
    indoor_avg = fac_long[fac_long["pollutant"] == "PM2.5"].groupby("facility")["value"].mean().reset_index()
    indoor_avg = indoor_avg.rename(columns={"value":"indoor_PM2.5"})
    # 전체 교실(모든 시설 평균)
    overall_indoor = indoor_avg["indoor_PM2.5"].mean()

    # 비교 차트: 실외 vs 실내(전체) + 시설별 표시
    comp_df = pd.DataFrame({
        "location": [f"실외: {country_choice}", "실내 평균(예시)"] + indoor_avg["facility"].tolist(),
        "PM2.5": [outdoor_val, overall_indoor] + indoor_avg["indoor_PM2.5"].round(2).tolist()
    })

    fig_comp = px.bar(comp_df, x="location", y="PM2.5", title=f"{country_choice} ({year_choice_comp}) 외부 vs 실내 PM2.5 비교", labels={"PM2.5":"PM2.5 (µg/m³)", "location":"구분"})
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("실내 측정/점검 현황 (연도별)")
    management_gap_df = datasets["management_gap_df"].copy()
    management_gap_df = remove_future_dates(management_gap_df, date_col="date")
    if not management_gap_df.empty:
        fig_m = px.bar(management_gap_df, x=management_gap_df["date"].dt.year.astype(str), y="value", labels={"x":"연도","value":"점검 비율 (%)"}, title="연도별 실내 공기질 측정/점검 비율")
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("점검 현황 데이터가 없습니다.")

    st.download_button("실내/실외 비교용 데이터 다운로드 (CSV)", data=comp_df.to_csv(index=False).encode("utf-8"), file_name="indoor_outdoor_comparison.csv", mime="text/csv")

# ---------- 탭2: 보고서(종합) ----------
with tabs[2]:
    st.header("종합 보고서: 실내·외 공기질과 청소년 건강")
    st.markdown("### 참고 자료")
    st.markdown("- 실내 공기질: WHO Indoor Air Quality Guidelines")
    st.markdown("- 실외 대기: 한국환경공단 에어코리아, IQAir World Air Quality Report")
    st.markdown("- 건강 영향: 대한소아청소년과학회, WHO GBD")

    st.markdown("### 분석 방법")
    st.markdown("- 측정 항목: 실내 CO2, PM2.5, TVOC / 실외 PM2.5")
    st.markdown("- 분석 방법: 평균 농도 비교, 시간대별 패턴, 상관관계 분석")
    st.markdown("- 건강 영향: 설문(두통, 집중력 저하, 피로감)과 연계")

    st.markdown("### 주요 발견 (요약)")
    st.markdown("- 점심 이후 교실 CO2 1,200ppm 이상 기록: 집중력 저하와 연관 가능성 존재")
    st.markdown("- 일부 학원/가정의 PM2.5가 35µg/m³ 이상으로 WHO 권고치 초과")
    st.markdown("- 외부 PM2.5가 높을 때 실내 공기질도 악화되는 상관성 관찰(예시)")

    st.markdown("### 실질적 제언(요약)")
    st.markdown("- 쉬는 시간 환기(2~3분), 공기청정기 필터 관리, 실내 흡연 금지 등 일상적 관리 권고")
    st.markdown("- 학급 단위 캠페인, 데이터 기록 및 시각화로 정책 제안 준비 권고")

    st.download_button("종합 보고서 요약(텍스트) 다운로드", data=("종합 보고서 요약\n"+"- 점심 이후 CO2 증가\n- 일부 공간 PM2.5 초과\n- 환기 권고").encode("utf-8"), file_name="report_summary.txt", mime="text/plain")

# ---------- 탭3: 예방 방법 및 계산기 ----------
with tabs[3]:
    st.header("예방 방법 및 실습형 도구들")
    st.markdown("아래 도구들은 간단한 가정 모델을 사용한 예시입니다. 실제 건강 영향은 개인·환경에 따라 다를 수 있습니다.")

    st.subheader("건강·학습 효과 계산기")
    st.markdown("교실 PM2.5 수준이 개선되었을 때 두통 발생률(예상) 변화 등을 간단히 추정합니다.")
    baseline_pm = st.number_input("현재 교실 PM2.5 (µg/m³)", min_value=0.0, value=35.0)
    improved_pm = st.number_input("환기/청정기로 개선된 예상 PM2.5 (µg/m³)", min_value=0.0, value=15.0)
    baseline_headache = st.slider("현재 두통 발생률(%) (교내 설문 예시)", 0, 100, 20)
    # 간단 모델: 1 µg/m3 PM2.5 감소당 두통율 0.4% 포인트 감소(예시)
    reduction_per_ug = 0.4
    delta = max(0, baseline_pm - improved_pm)
    estimated_reduction = round(min(baseline_headache, delta * reduction_per_ug), 2)
    est_headache_after = round(max(0, baseline_headache - estimated_reduction), 2)
    st.write(f"예상 두통 발생률 감소: {estimated_reduction} %p → 개선 후 약 {est_headache_after} %")

    st.markdown("---")
    st.subheader("오늘의 교실 공기질 지수 알리미")
    st.markdown("외부 PM2.5와 교실 CO2, PM2.5(측정값)를 비교하여 간단한 행동가이드를 제시합니다.")
    out_pm = st.number_input("오늘 실외 PM2.5 (µg/m³)", min_value=0.0, value=30.0, key="out_pm")
    in_pm = st.number_input("오늘 교실 PM2.5 (µg/m³)", min_value=0.0, value=40.0, key="in_pm")
    in_co2 = st.number_input("오늘 교실 CO2 (ppm)", min_value=200, value=1200, key="in_co2")

    # 간단 행동 가이드
    guide = "정상"
    if in_pm > 35 or out_pm > 75:
        guide = "환기 필수 🔄"
    if in_co2 > 1200:
        guide = "즉시 환기 권장 (CO2 높음)"
    if in_pm > 75 and out_pm > 150:
        guide = "실내 대피(마스크 착용) 권고"
    st.metric("오늘 행동 가이드", guide)

    st.markdown("---")
    st.subheader("교실 식물 효과 계산기")
    st.markdown("간단한 흡수 계수를 사용해 식물의 CO2 흡수량·습도 영향 추정")
    plant_options = {"스파티필름": 0.05, "아레카야자": 0.08, "몬스테라": 0.06}  # kg CO2/day(예시)
    plant_choice = st.selectbox("식물 종류", list(plant_options.keys()))
    plant_count = st.number_input("식물 개수", min_value=0, value=3)
    co2_absorb = plant_options[plant_choice] * plant_count
    est_humidity = min(5, 0.5 * plant_count)  # 습도(%) 변화 예상(예시)
    st.write(f"예상 일일 CO2 흡수량(가정): {co2_absorb:.2f} kg/day, 예상 습도 개선: {est_humidity:.1f}%")

    st.markdown("---")
    st.subheader("폐 건강 위험 예측기 (간단 모델)")
    st.markdown("입력값에 따라 호흡기 자극 위험 수준을 추정합니다. 실제 진단이 아님을 유의하세요.")
    in_out_pm = st.number_input("실외 PM2.5 (µg/m³)", min_value=0.0, value=30.0, key="lung_out")
    vent_freq = st.selectbox("일일 환기 횟수", ["거의 없음","하루 1회","하루 2-3회","자주(>3회)"])
    mask_use = st.selectbox("평균 마스크 착용률", ["낮음","보통","높음"]) 
    score = 0
    score += 0 if in_out_pm < 15 else (1 if in_out_pm < 35 else 2 if in_out_pm < 75 else 3)
    score += 0 if vent_freq=="자주(>3회)" else (1 if vent_freq=="하루 2-3회" else 2 if vent_freq=="하루 1회" else 3)
    score -= 1 if mask_use=="높음" else (0 if mask_use=="보통" else 1)
    if score <= 1:
        risk = "낮음"
    elif score == 2 or score == 3:
        risk = "보통"
    else:
        risk = "높음"
    st.write(f"예상 호흡기 자극 위험 수준: {risk}")

    st.markdown("---")
    st.subheader("실내 대기질 예방 체크리스트")
    checklist_items = [
        "창문 열고 환기하기 (하루 2~3번, 짧게)",
        "공기청정기 사용하기 (필터 정기 점검)",
        "바닥·가구 먼지 자주 청소하기",
        "침구·커튼 정기 세탁하기",
        "향초·스프레이형 방향제 줄이기",
        "적정 습도 유지하기 (40~60%)",
        "곰팡이 관리(환기+제습)",
        "공기 정화 식물 배치",
        "친환경 세제 사용",
        "반려동물 관리(털 제거 등)"
    ]
    checked = 0
    cols = st.columns(2)
    for i, item in enumerate(checklist_items):
        c = cols[i%2]
        if c.checkbox(item, key=f"chk_{i}"):
            checked += 1
    progress = int(checked / len(checklist_items) * 100)
    st.progress(progress)
    if progress == 100:
        st.success("축하합니다! 예방 체크리스트를 모두 완료하셨습니다 🎉")

# ---------- 탭4: 제언 및 행동 ----------
with tabs[4]:
    st.header("제언 및 행동")
    st.markdown("데이터 기반의 시민 행동과 학교 단위 제안을 정리합니다.")
    st.markdown("- 교실별 공기질 기록 및 정기 보고 시행")
    st.markdown("- 학생회 주도의 환기·공기질 캠페인 전개")
    st.markdown("- 교육청에 공기질 개선 제안서 제출(필터 교체 주기 등)")
    st.markdown("- 정책과 예산(주민참여예산 등)을 통한 실질적 개선 시도")

    st.markdown("---")
    st.markdown("※ 주의: 본 대시보드는 제공된 보고서 텍스트를 바탕으로 생성된 예시 데이터와 간단 모델을 사용합니다. 실제 정책·의료 판단에는 WHO, 한국환경공단 등의 공식 데이터를 참고하세요.")

# EOF