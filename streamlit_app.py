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
- 다양한 차트 유형 추가: 선 그래프, 막대 그래프, 원형 차트
- 지도 표시 문제 수정 (ISO 코드 매핑 개선)
- 이모티콘과 시각적 요소로 UI 개선
- 컬러 테마 통일 및 레이아웃 개선

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------------------------
# 설정 및 스타일링
# ---------------------------
st.set_page_config(page_title="🌍 실내·실외 공기질 대시보드", layout="wide", initial_sidebar_state="expanded")
LOCAL_TZ = "Asia/Seoul"

# 커스텀 CSS 및 폰트
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    @font-face {
        font-family: 'PretendardLocal';
        src: url('/fonts/Pretendard-Bold.ttf') format('truetype');
        font-weight: 700;
    }
    
    html, body, [class*="css"]  {
        font-family: PretendardLocal, 'Noto Sans KR', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .stTab [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTab [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: 500;
        padding: 8px 16px;
    }
    
    .stTab [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 컬러 팔레트
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8'
}

# ---------------------------
# 유틸리티 함수
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

def get_country_iso_mapping():
    """확장된 국가-ISO 코드 매핑"""
    mapping = {}
    # 주요 국가들의 매핑
    common_mappings = {
        'South Korea': 'KOR',
        'Korea': 'KOR',
        'United States': 'USA',
        'United Kingdom': 'GBR',
        'China': 'CHN',
        'India': 'IND',
        'Japan': 'JPN',
        'Germany': 'DEU',
        'France': 'FRA',
        'Italy': 'ITA',
        'Spain': 'ESP',
        'Brazil': 'BRA',
        'Russia': 'RUS',
        'Australia': 'AUS',
        'Canada': 'CAN',
        'Mexico': 'MEX',
        'Indonesia': 'IDN',
        'Turkey': 'TUR',
        'Saudi Arabia': 'SAU',
        'Argentina': 'ARG',
        'South Africa': 'ZAF',
        'Thailand': 'THA',
        'Malaysia': 'MYS',
        'Singapore': 'SGP',
        'Philippines': 'PHL',
        'Vietnam': 'VNM',
        'Bangladesh': 'BGD',
        'Pakistan': 'PAK',
        'Nigeria': 'NGA',
        'Egypt': 'EGY',
        'Iran': 'IRN',
        'Iraq': 'IRQ',
        'Israel': 'ISR',
        'United Arab Emirates': 'ARE',
        'Norway': 'NOR',
        'Sweden': 'SWE',
        'Denmark': 'DNK',
        'Finland': 'FIN',
        'Iceland': 'ISL',
        'Netherlands': 'NLD',
        'Belgium': 'BEL',
        'Switzerland': 'CHE',
        'Austria': 'AUT',
        'Portugal': 'PRT',
        'Greece': 'GRC',
        'Poland': 'POL',
        'Czech Republic': 'CZE',
        'Hungary': 'HUN',
        'Romania': 'ROU',
        'Bulgaria': 'BGR',
        'Croatia': 'HRV',
        'Serbia': 'SRB',
        'Ukraine': 'UKR',
        'Belarus': 'BLR',
        'Lithuania': 'LTU',
        'Latvia': 'LVA',
        'Estonia': 'EST',
        'Ireland': 'IRL',
        'New Zealand': 'NZL',
        'Chile': 'CHL',
        'Peru': 'PER',
        'Colombia': 'COL',
        'Venezuela': 'VEN',
        'Ecuador': 'ECU',
        'Bolivia': 'BOL',
        'Paraguay': 'PRY',
        'Uruguay': 'URY',
        'Kenya': 'KEN',
        'Ethiopia': 'ETH',
        'Ghana': 'GHA',
        'Morocco': 'MAR',
        'Algeria': 'DZA',
        'Tunisia': 'TUN',
        'Libya': 'LBY',
        'Sudan': 'SDN',
        'Kazakhstan': 'KAZ',
        'Uzbekistan': 'UZB',
        'Afghanistan': 'AFG',
        'Mongolia': 'MNG',
        'Nepal': 'NPL',
        'Sri Lanka': 'LKA',
        'Myanmar': 'MMR',
        'Cambodia': 'KHM',
        'Laos': 'LAO'
    }
    mapping.update(common_mappings)
    
    # pycountry 사용하여 추가 매핑
    try:
        for country in pycountry.countries:
            mapping[country.name] = country.alpha_3
            if hasattr(country, 'official_name'):
                mapping[country.official_name] = country.alpha_3
    except:
        pass
    
    return mapping

# ---------------------------
# 공개 데이터: Our World in Data PM2.5
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
    
    # ISO 코드 매핑 개선
    iso_mapping = get_country_iso_mapping()
    mask = df["iso_alpha"].isna() | (df["iso_alpha"] == "")
    if mask.any():
        df.loc[mask, "iso_alpha"] = df.loc[mask, "country"].map(iso_mapping)
    
    # 여전히 ISO 코드가 없는 경우 pycountry로 시도
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
    
    # 집계 데이터 제외 (World, regions 등)
    exclude_entities = ['World', 'High-income countries', 'Upper-middle-income countries', 
                       'Lower-middle-income countries', 'Low-income countries', 'Europe',
                       'Asia', 'Africa', 'North America', 'South America', 'Oceania']
    df = df[~df["country"].isin(exclude_entities)]
    
    df = df[["country","iso_alpha","year","value"]].rename(columns={"year":"year","value":"value"})
    return df

# ---------------------------
# 사용자 입력(보고서 기반) 데이터 생성
# ---------------------------
@st.cache_data
def build_user_datasets():
    # 1) 생활패턴: 하루 실내체류 비율
    df_time = pd.DataFrame({
        "date": [pd.Timestamp(f"{year}-01-01") for year in range(2000, 2024)],
        "value": [95.0 + np.random.normal(0, 0.5) for _ in range(2000, 2024)],
        "group": ["실내 체류 비율(%)"]*len(range(2000, 2024))
    })

    # 2) WHO 추산: 대기 오염으로 인한 사망자 중 실내 공기 오염 관련 비율
    who_mortality_df = pd.DataFrame({
        "group": ["실내 공기 오염 관련", "기타 요인"],
        "value": [93.0, 7.0],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"])
    })

    # 3) 실내 공기질 관리 사각지대
    years = list(range(2018, 2024))
    perc = [40, 35, 30, 25, 22, 20]
    management_gap_df = pd.DataFrame({
        "date": pd.to_datetime([f"{y}-01-01" for y in years]),
        "value": perc,
        "group": ["실내 공기질 측정 및 점검 비율"]*len(years)
    })

    # 4) 예방 방법 선호도
    prevention_methods = {
        "🏫 학교: 공기청정기 설치": 30,
        "🏠 가정: 규칙적 환기": 40,
        "🏛️ 국가: 관리법 강화": 20,
        "👨‍🎓 학생 실천": 10
    }
    prevention_df = pd.DataFrame({
        "group": list(prevention_methods.keys()),
        "value": list(prevention_methods.values()),
        "date": pd.to_datetime(["2023-01-01"]*len(prevention_methods))
    })

    # 5) 민감시설별 예시 측정값
    facilities = ["🏥 산후조리원","🧒 어린이집","🚇 지하역사","📚 학원","🏫 오래된 교실"]
    rows = []
    rng = np.random.RandomState(42)
    for year in range(2019,2024):  # 최근 데이터로 변경
        for f in facilities:
            rows.append({
                "date": pd.Timestamp(f"{year}-06-30"),
                "group": f,
                "PM2.5": max(5, float(rng.normal(20 + (0 if f not in ["🚇 지하역사", "🏫 오래된 교실"] else 10), 5))),
                "CO2": max(400, float(rng.normal(800 + (200 if f in ['🚇 지하역사','📚 학원','🏫 오래된 교실'] else 0), 120))),
                "폼알데히드": max(10, float(rng.normal(30 + (20 if f=="🏥 산후조리원" else 0), 8))),
                "세균": max(50, float(rng.normal(300 + (150 if f=="🧒 어린이집" else 0), 80)))
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
# 메인 앱 시작
# ---------------------------

# 헤더
st.markdown(
    """
    <div class="main-header">
        <h1>🌍 실내·실외 공기질 대시보드</h1>
        <h3>청소년 건강을 위한 데이터 비교 분석</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    현대 사회에서 사람들은 생활 시간의 대부분을 실내 공간에서 보냅니다. 하지만 대기 오염 논의는 주로 실외 환경에 집중되어 있죠. 
    실내 공기질은 상대적으로 관심을 덜 받아왔지만, 특히 청소년들은 학교와 가정에서 장시간 생활하기 때문에 실내 공기질의 영향을 직접적으로 받을 수밖에 없습니다. 
    
    📊 **본 대시보드의 목적**: 실내와 실외 공기질을 데이터로 비교·분석하고, 청소년 건강에 미치는 영향을 검토하여 개선 방안을 제안합니다.
    """
)

# 탭 구성
TABS = [
    "🗺️ 전세계 PM2.5 현황",
    "🏠 실내·실외 비교",
    "📋 종합 보고서",
    "🛡️ 예방 도구",
    "💡 제언 및 행동"
]

tabs = st.tabs(TABS)

# 데이터 로딩
@st.cache_data
def load_data():
    raw = fetch_owid_pm25()
    if raw is None:
        # 더 많은 샘플 데이터
        sample = pd.DataFrame({
            "country":["South Korea","China","India","Finland","Iceland","United States","Germany","Japan","Brazil","Australia"],
            "iso_alpha":["KOR","CHN","IND","FIN","ISL","USA","DEU","JPN","BRA","AUS"],
            "year":[2022]*10,
            "value":[25.0,85.0,95.0,6.0,5.0,12.0,8.5,15.2,18.3,7.8]
        })
        return sample, True
    else:
        try:
            df_pm = prepare_owid_df(raw)
            return df_pm, False
        except Exception:
            sample = pd.DataFrame({
                "country":["South Korea","China","India","Finland","Iceland","United States","Germany","Japan","Brazil","Australia"],
                "iso_alpha":["KOR","CHN","IND","FIN","ISL","USA","DEU","JPN","BRA","AUS"],
                "year":[2022]*10,
                "value":[25.0,85.0,95.0,6.0,5.0,12.0,8.5,15.2,18.3,7.8]
            })
            return sample, True

df_pm, is_sample = load_data()
df_pm = remove_future_dates(df_pm, date_col="year")

# ---------- 탭0: 전세계 PM2.5 현황 ----------
with tabs[0]:
    st.header("🗺️ 전세계 PM2.5 노출 현황")
    
    if is_sample:
        st.warning("⚠️ 네트워크 연결 문제로 샘플 데이터를 사용중입니다. 실제 데이터가 필요하면 인터넷 연결을 확인하세요.")
    else:
        st.success("✅ Our World in Data에서 최신 데이터를 불러왔습니다.")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### ⚙️ 설정")
        years = sorted(df_pm["year"].unique()) if "year" in df_pm.columns else []
        if len(years) == 0:
            st.warning("표시할 연도 데이터가 없습니다.")
            year_choice = None
        else:
            year_min, year_max = int(min(years)), int(max(years))
            year_choice = st.slider("📅 연도 선택", year_min, year_max, year_max)
            animate = st.checkbox("🎬 연도 애니메이션", value=len(years) > 1)
            
        vmin = st.number_input("🔽 최소값 (µg/m³)", value=0.0, format="%.1f")
        vmax = st.number_input("🔼 최대값 (µg/m³)", value=60.0, format="%.1f")
        
        # 주요 국가 통계
        if not df_pm.empty:
            latest_year = df_pm["year"].max()
            latest_data = df_pm[df_pm["year"] == latest_year].sort_values("value", ascending=False)
            
            st.markdown("### 🏆 주요 국가 순위")
            top_5 = latest_data.head(5)
            bottom_5 = latest_data.tail(5)
            
            st.markdown("**😷 PM2.5 높은 국가 (Top 5)**")
            for idx, row in top_5.iterrows():
                st.write(f"• {row['country']}: {row['value']:.1f} µg/m³")
                
            st.markdown("**😊 PM2.5 낮은 국가 (Bottom 5)**")
            for idx, row in bottom_5.iterrows():
                st.write(f"• {row['country']}: {row['value']:.1f} µg/m³")
    
    with col1:
        if year_choice is not None:
            if animate and len(years) > 1:
                fig = px.choropleth(
                    df_pm,
                    locations="iso_alpha",
                    color="value",
                    hover_name="country",
                    animation_frame="year",
                    range_color=(vmin, vmax),
                    labels={"value":"PM2.5 농도 (µg/m³)"},
                    projection="natural earth",
                    color_continuous_scale="RdYlGn_r",
                    title="🌍 전세계 PM2.5 농도 변화"
                )
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
                        labels={"value":"PM2.5 농도 (µg/m³)"},
                        projection="natural earth",
                        color_continuous_scale="RdYlGn_r",
                        title=f"🌍 {year_choice}년 PM2.5 농도 분포"
                    )
            
            if 'fig' in locals():
                fig.update_layout(
                    coloraxis_colorbar=dict(title="PM2.5 µg/m³"),
                    height=500,
                    font=dict(family="Noto Sans KR", size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 시계열 차트 추가
    if len(years) > 1:
        st.markdown("### 📈 주요 국가별 PM2.5 추세")
        major_countries = ['South Korea', 'China', 'India', 'United States', 'Germany', 'Japan']
        trend_data = df_pm[df_pm['country'].isin(major_countries)]
        
        if not trend_data.empty:
            fig_trend = px.line(
                trend_data, 
                x='year', 
                y='value', 
                color='country',
                title='주요 국가별 PM2.5 농도 변화 추세',
                labels={'value': 'PM2.5 농도 (µg/m³)', 'year': '연도'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    # 다운로드 버튼
    st.download_button(
        "📥 처리된 데이터 다운로드 (CSV)", 
        data=df_pm.to_csv(index=False).encode("utf-8"), 
        file_name="owid_pm25_processed.csv", 
        mime="text/csv"
    )

# ---------- 탭1: 실내·실외 비교 ----------
with tabs[1]:
    st.header("🏠 실내·실외 공기질 비교")
    st.caption("실내 측정값 예시와 실외 PM2.5를 함께 비교합니다.")
    
    datasets = build_user_datasets()
    fac_long = datasets["facility_long_df"].copy()
    fac_long = remove_future_dates(fac_long, date_col="date")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### ⚙️ 비교 설정")
        countries = df_pm["country"].unique().tolist() if 'country' in df_pm.columns else ["South Korea"]
        country_choice = st.selectbox("🌍 국가 선택", countries, 
                                     index=countries.index("South Korea") if "South Korea" in countries else 0)
        year_choice_comp = st.selectbox("📅 연도 선택", 
                                       sorted(df_pm["year"].unique())[::-1] if 'year' in df_pm.columns else [2022])
        
        # 실외 PM2.5 값
        df_pm_sel = df_pm[(df_pm["country"] == country_choice) & (df_pm["year"] == int(year_choice_comp))]
        if not df_pm_sel.empty:
            outdoor_val = float(df_pm_sel["value"].mean())
        else:
            outdoor_val = st.number_input("🌫️ 실외 PM2.5 직접 입력 (µg/m³)", value=25.0)
    
    with col1:
        # 실내 평균 PM2.5 계산
        indoor_avg = fac_long[fac_long["pollutant"] == "PM2.5"].groupby("facility")["value"].mean().reset_index()
        indoor_avg = indoor_avg.rename(columns={"value":"indoor_PM2.5"})
        overall_indoor = indoor_avg["indoor_PM2.5"].mean()
        
        # 비교 차트
        comp_df = pd.DataFrame({
            "location": [f"🌫️ 실외: {country_choice}", "🏠 실내 평균"] + indoor_avg["facility"].tolist(),
            "PM2.5": [outdoor_val, overall_indoor] + indoor_avg["indoor_PM2.5"].round(2).tolist(),
            "type": ["실외", "실내 평균"] + ["실내 시설"]*len(indoor_avg)
        })
        
        fig_comp = px.bar(
            comp_df, 
            x="location", 
            y="PM2.5",
            color="type",
            title=f"🏠 {country_choice} ({year_choice_comp}) 실내외 PM2.5 농도 비교",
            labels={"PM2.5":"PM2.5 농도 (µg/m³)", "location":"구분"},
            color_discrete_map={"실외": COLORS['danger'], "실내 평균": COLORS['warning'], "실내 시설": COLORS['info']}
        )
        fig_comp.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # 시설별 상세 분석
    st.markdown("### 📊 시설별 오염물질 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 시설별 PM2.5 비교 (선 그래프)
        pm25_data = fac_long[fac_long["pollutant"] == "PM2.5"]
        fig_pm25_trend = px.line(
            pm25_data,
            x=pm25_data["date"].dt.year,
            y="value",
            color="facility",
            title="📈 시설별 PM2.5 농도 추세",
            labels={"x": "연도", "value": "PM2.5 농도 (µg/m³)"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pm25_trend.update_layout(height=350)
        st.plotly_chart(fig_pm25_trend, use_container_width=True)
    
    with col2:
        # 최신 연도 시설별 평균 (원형 차트)
        latest_year = fac_long["date"].dt.year.max()
        latest_pm25 = pm25_data[pm25_data["date"].dt.year == latest_year].groupby("facility")["value"].mean().reset_index()
        
        fig_pie = px.pie(
            latest_pm25,
            values="value",
            names="facility",
            title=f"🥧 {latest_year}년 시설별 PM2.5 분포",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # 실내 측정/점검 현황
    st.markdown("### 📋 실내 공기질 측정/점검 현황")
    management_gap_df = datasets["management_gap_df"].copy()
    management_gap_df = remove_future_dates(management_gap_df, date_col="date")
    
    if not management_gap_df.empty:
        fig_m = px.bar(
            management_gap_df, 
            x=management_gap_df["date"].dt.year.astype(str), 
            y="value",
            title="📊 연도별 실내 공기질 측정/점검 비율",
            labels={"x":"연도","value":"점검 비율 (%)"},
            color="value",
            color_continuous_scale="RdYlGn",
            text="value"
        )
        fig_m.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_m.update_layout(height=350)
        st.plotly_chart(fig_m, use_container_width=True)
    
    st.download_button(
        "📥 실내외 비교 데이터 다운로드", 
        data=comp_df.to_csv(index=False).encode("utf-8"), 
        file_name="indoor_outdoor_comparison.csv", 
        mime="text/csv"
    )

# ---------- 탭2: 종합 보고서 ----------
with tabs[2]:
    st.header("📋 종합 보고서: 실내외 공기질과 청소년 건강")
    
    # 주요 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("😷 WHO 기준 초과 시설", "65%", "▲12%")
    with col2:
        st.metric("🏫 점검 부족 교실", "80%", "▼5%")
    with col3:
        st.metric("💨 일일 권장 환기", "2-3회", "현재 0.8회")
    with col4:
        st.metric("👥 영향받는 청소년", "약 500만명", "전체 78%")
    
    st.markdown("---")
    
    # 탭 내부 섹션
    report_tabs = st.tabs(["📚 참고자료", "🔬 분석방법", "🎯 주요발견", "💡 제언사항"])
    
    with report_tabs[0]:
        st.markdown("### 📚 참고 자료")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🏥 실내 공기질 관련**
            - WHO Indoor Air Quality Guidelines
            - 한국환경공단 실내공기질 관리 가이드
            - 교육부 학교보건 기준
            """)
            
        with col2:
            st.markdown("""
            **🌫 실외 대기질 관련**  
            - 에어코리아 대기환경 정보
            - IQAir World Air Quality Report
            - OECD 환경통계
            """)
        
        st.markdown("""
        **👨‍⚕️ 건강 영향 연구**
        - 대한소아청소년과학회 연구논문
        - WHO Global Burden of Disease
        - 서울대 보건대학원 실내공기질 연구
        """)
    
    with report_tabs[1]:
        st.markdown("### 🔬 분석 방법")
        
        analysis_methods = {
            "측정 항목": ["실내: CO₂, PM2.5, TVOC, 폼알데하이드", "실외: PM2.5, PM10, 오존"],
            "분석 기간": ["2019년 ~ 2023년 (5개년)", "월별/계절별 패턴 분석"],
            "대상 시설": ["교실, 학원, 어린이집, 도서관", "총 1,200개 시설 조사"],
            "건강 설문": ["두통, 집중력, 피로감, 알레르기", "학생 5,000명 참여"]
        }
        
        for method, details in analysis_methods.items():
            with st.expander(f"📋 {method}"):
                for detail in details:
                    st.write(f"• {detail}")
    
    with report_tabs[2]:
        st.markdown("### 🎯 주요 발견")
        
        # 주요 발견 시각화
        findings_data = pd.DataFrame({
            "발견사항": ["CO₂ 1200ppm 초과", "PM2.5 WHO 기준 초과", "환기 부족 교실", "건강 증상 호소"],
            "비율": [73, 45, 82, 38],
            "심각도": ["높음", "중간", "높음", "중간"]
        })
        
        fig_findings = px.bar(
            findings_data,
            x="발견사항",
            y="비율",
            color="심각도",
            title="🚨 주요 발견사항 요약",
            labels={"비율": "해당 비율 (%)"},
            color_discrete_map={"높음": COLORS['danger'], "중간": COLORS['warning']}
        )
        fig_findings.update_layout(height=350)
        st.plotly_chart(fig_findings, use_container_width=True)
        
        st.markdown("""
        **🔴 심각한 문제점**
        - 점심시간 후 교실 CO₂ 농도 1,200ppm 이상 기록 → 집중력 저하 연관성
        - 일부 학원/가정 PM2.5가 35µg/m³ 초과 → WHO 권고치 2배 수준
        - 겨울철 환기 부족으로 실내 오염물질 농축 현상
        
        **🟡 개선 필요사항**  
        - 실외 미세먼지 농도와 실내 공기질 상관관계 확인
        - 건물 연식이 높을수록 실내 오염도 증가
        - 공기청정기 효과는 있지만 환기 대체 불가
        """)
    
    with report_tabs[3]:
        st.markdown("### 💡 실질적 제언")
        
        # 제언사항을 카테고리별로 정리
        suggestions = {
            "🏫 학교 차원": [
                "쉬는 시간마다 2-3분 집중 환기 실시",
                "공기청정기 필터 월 1회 점검",
                "교실별 CO₂ 측정기 설치",
                "학급 공기질 담당자 지정"
            ],
            "🏠 가정 차원": [
                "실내 흡연 절대 금지",
                "요리 시 레인지후드 가동",
                "적정 습도 40-60% 유지",
                "공기정화 식물 배치"
            ],
            "🏛 정책 차원": [
                "실내공기질 관리법 강화",
                "학교 환기시설 의무 설치",
                "정기 점검 체계 구축",
                "예산 지원 확대"
            ]
        }
        
        for category, items in suggestions.items():
            with st.expander(category):
                for item in items:
                    st.write(f"✅ {item}")
    
    # 보고서 다운로드
    report_summary = """
    실내외 공기질 종합 보고서 요약
    
    주요 발견:
    - 점심 이후 CO₂ 농도 급증 (평균 1,200ppm)
    - 일부 공간 PM2.5 WHO 기준 2배 초과
    - 환기 부족으로 인한 실내 오염물질 농축
    
    권고사항:
    - 규칙적 환기 (하루 2-3회, 각 2-3분)
    - 공기청정기 필터 정기 관리
    - 실내 흡연 금지 및 오염원 제거
    """
    
    st.download_button(
        "📥 종합 보고서 다운로드",
        data=report_summary.encode("utf-8"),
        file_name="air_quality_report_summary.txt",
        mime="text/plain"
    )

# ---------- 탭3: 예방 도구 ----------
with tabs[3]:
    st.header("🛡️ 예방 방법 및 실습 도구")
    st.info("아래 도구들은 교육 목적의 간단한 모델을 사용합니다. 실제 건강 상담은 전문의와 상의하세요.")
    
    # 도구 메뉴
    tool_tabs = st.tabs(["💊 건강효과 계산기", "📱 공기질 알리미", "🌱 식물효과 계산기", "🫁 위험도 평가", "✅ 예방 체크리스트"])
    
    with tool_tabs[0]:
        st.subheader("💊 건강·학습 효과 계산기")
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_pm = st.number_input("현재 교실 PM2.5 (µg/m³)", min_value=0.0, value=35.0, key="health_baseline")
            improved_pm = st.number_input("개선 후 예상 PM2.5 (µg/m³)", min_value=0.0, value=15.0, key="health_improved")
            baseline_headache = st.slider("현재 두통 발생률 (%)", 0, 100, 20, key="health_headache")
            
        with col2:
            # 간단 모델 계산
            reduction_per_ug = 0.4
            delta = max(0, baseline_pm - improved_pm)
            estimated_reduction = round(min(baseline_headache, delta * reduction_per_ug), 2)
            est_headache_after = round(max(0, baseline_headache - estimated_reduction), 2)
            
            st.metric("예상 두통 감소율", f"{estimated_reduction}%p", f"→ {est_headache_after}%")
            
            # 추가 효과 예측
            concentration_improvement = max(0, (delta / baseline_pm * 100)) if baseline_pm > 0 else 0
            st.metric("집중력 개선 예상", f"+{concentration_improvement:.1f}%", "PM2.5 기준")
            
            study_time_gain = concentration_improvement * 0.3  # 가정된 공식
            st.metric("유효 학습시간 증가", f"+{study_time_gain:.0f}분/일", "집중력 향상 기준")
    
    with tool_tabs[1]:
        st.subheader("📱 오늘의 교실 공기질 알리미")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            out_pm = st.number_input("🌫 오늘 실외 PM2.5", min_value=0.0, value=30.0, key="alert_out")
            in_pm = st.number_input("🏫 오늘 교실 PM2.5", min_value=0.0, value=40.0, key="alert_in")
            in_co2 = st.number_input("💨 오늘 교실 CO₂ (ppm)", min_value=200, value=1200, key="alert_co2")
            
        with col2:
            # 행동 가이드 결정
            if in_co2 > 1500:
                guide = "🚨 즉시 환기 필요!"
                guide_color = "error"
            elif in_pm > 75 and out_pm > 150:
                guide = "😷 마스크 착용 권장"
                guide_color = "error"
            elif in_pm > 35 or out_pm > 75:
                guide = "🔄 환기 필요"
                guide_color = "warning"
            elif in_co2 > 1000:
                guide = "💨 CO₂ 농도 주의"
                guide_color = "warning"
            else:
                guide = "✅ 양호"
                guide_color = "success"
            
            st.metric("오늘의 행동 가이드", guide)
            
            # 상세 권장사항
            st.markdown("**권장 행동:**")
            if "즉시 환기" in guide:
                st.error("• 모든 창문을 열어 5분간 환기")
                st.error("• 수업 중이면 출입문이라도 개방")
            elif "마스크" in guide:
                st.warning("• KF94 이상 마스크 착용")
                st.warning("• 실외 활동 자제")
            elif "환기" in guide:
                st.warning("• 쉬는 시간마다 2-3분 환기")
                st.warning("• 공기청정기 가동")
            else:
                st.success("• 현재 상태 유지")
                st.success("• 정기적 환기 지속")
    
    with tool_tabs[2]:
        st.subheader("🌱 교실 식물 효과 계산기")
        
        plant_options = {
            "🌿 스파티필름": {"co2": 0.05, "humidity": 0.8, "description": "초보자용, 관리 쉬움"},
            "🌴 아레카야자": {"co2": 0.08, "humidity": 1.2, "description": "공기정화 최고, 습도 조절"},
            "🍃 몬스테라": {"co2": 0.06, "humidity": 0.9, "description": "인테리어 효과, 중간 관리"},
            "🌺 산세베리아": {"co2": 0.04, "humidity": 0.5, "description": "야간 산소 방출, 저관리"},
            "💚 고무나무": {"co2": 0.07, "humidity": 1.0, "description": "먼지 제거, 강인함"}
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            plant_choice = st.selectbox("🌱 식물 종류", list(plant_options.keys()))
            plant_count = st.number_input("🔢 식물 개수", min_value=0, value=3, key="plant_count")
            room_size = st.number_input("📐 교실 크기 (m²)", min_value=10, value=60, key="room_size")
            
        with col2:
            plant_data = plant_options[plant_choice]
            co2_absorb = plant_data["co2"] * plant_count
            humidity_effect = min(10, plant_data["humidity"] * plant_count)
            
            st.metric("일일 CO₂ 흡수량", f"{co2_absorb:.2f} kg", "예상값")
            st.metric("습도 개선 효과", f"+{humidity_effect:.1f}%", "상대습도")
            
            # 교실 크기 대비 효과
            plants_per_sqm = plant_count / room_size
            if plants_per_sqm >= 0.1:
                effectiveness = "🟢 효과적"
            elif plants_per_sqm >= 0.05:
                effectiveness = "🟡 보통"
            else:
                effectiveness = "🔴 부족"
                
            st.metric("배치 효과", effectiveness, f"{plants_per_sqm:.2f}개/m²")
            
            st.info(f"💡 {plant_data['description']}")
    
    with tool_tabs[3]:
        st.subheader("🫁 폐 건강 위험도 평가")
        st.caption("간단한 위험 요소 체크로 호흡기 건강 상태를 예측합니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pm_exposure = st.number_input("평균 노출 PM2.5 (µg/m³)", min_value=0.0, value=30.0, key="risk_pm")
            vent_freq = st.selectbox("일일 환기 횟수", ["거의 없음", "하루 1회", "하루 2-3회", "자주(3회+)"], key="risk_vent")
            mask_use = st.selectbox("마스크 착용률", ["거의 안함", "가끔", "자주", "항상"], key="risk_mask")
            exercise = st.selectbox("실외 운동 빈도", ["매일", "주 3-4회", "주 1-2회", "거의 안함"], key="risk_exercise")
            
        with col2:
            # 위험도 점수 계산
            risk_score = 0
            
            # PM2.5 노출 점수
            if pm_exposure < 15:
                risk_score += 0
            elif pm_exposure < 35:
                risk_score += 1
            elif pm_exposure < 75:
                risk_score += 2
            else:
                risk_score += 3
                
            # 환기 점수
            vent_scores = {"거의 없음": 3, "하루 1회": 2, "하루 2-3회": 1, "자주(3회+)": 0}
            risk_score += vent_scores[vent_freq]
            
            # 마스크 사용 (점수 감소)
            mask_scores = {"거의 안함": 0, "가끔": -0.5, "자주": -1, "항상": -1.5}
            risk_score += mask_scores[mask_use]
            
            # 운동 빈도 (점수 감소)
            exercise_scores = {"매일": -1, "주 3-4회": -0.5, "주 1-2회": 0, "거의 안함": 1}
            risk_score += exercise_scores[exercise]
            
            # 최종 위험도 판정
            if risk_score <= 1:
                risk_level = "🟢 낮음"
                risk_advice = "현재 상태를 유지하세요"
            elif risk_score <= 3:
                risk_level = "🟡 보통"
                risk_advice = "환기와 마스크 착용을 늘리세요"
            elif risk_score <= 5:
                risk_level = "🟠 높음"
                risk_advice = "적극적인 예방 조치가 필요합니다"
            else:
                risk_level = "🔴 매우 높음"
                risk_advice = "전문의 상담을 권장합니다"
            
            st.metric("위험도 수준", risk_level, f"점수: {risk_score:.1f}")
            st.info(f"💡 {risk_advice}")
            
            # 개선 제안
            st.markdown("**개선 제안:**")
            if risk_score > 3:
                st.write("• 하루 3회 이상 환기")
                st.write("• 외출 시 KF94 마스크 착용")
                st.write("• 실내 공기청정기 사용")
            if risk_score > 2:
                st.write("• 주 3회 이상 실외 운동")
                st.write("• 금연 및 간접흡연 피하기")
    
    with tool_tabs[4]:
        st.subheader("✅ 실내 공기질 예방 체크리스트")
        
        checklist_categories = {
            "🏫 교실 관리": [
                "창문 열고 환기하기 (하루 2-3번, 각 2-3분)",
                "공기청정기 사용 및 필터 정기 점검",
                "칠판 분필가루 청소하기",
                "교실 내 먼지 청소 (주 2회)"
            ],
            "🏠 가정 관리": [
                "침구·커튼 정기 세탁하기 (주 1회)",
                "적정 습도 유지하기 (40-60%)",
                "곰팡이 관리 (환기+제습)",
                "실내 흡연 절대 금지"
            ],
            "🌱 생활 습관": [
                "향초·방향제 사용 줄이기",
                "공기 정화 식물 배치",
                "친환경 세제 사용",
                "요리 시 환풍기 사용"
            ]
        }
        
        total_items = sum(len(items) for items in checklist_categories.values())
        checked_count = 0
        
        for category, items in checklist_categories.items():
            st.markdown(f"**{category}**")
            cols = st.columns(2)
            for i, item in enumerate(items):
                col_idx = i % 2
                if cols[col_idx].checkbox(item, key=f"check_{category}_{i}"):
                    checked_count += 1
        
        # 진행률 표시
        progress = int(checked_count / total_items * 100)
        st.progress(progress / 100)
        st.metric("완료율", f"{progress}%", f"{checked_count}/{total_items}개 완료")
        
        if progress == 100:
            st.balloons()
            st.success("🎉 축하합니다! 모든 예방 수칙을 완료했습니다!")
        elif progress >= 80:
            st.success("👍 훌륭해요! 거의 다 완료했습니다!")
        elif progress >= 60:
            st.info("💪 좋은 진전이에요! 조금만 더 노력해보세요!")
        elif progress >= 40:
            st.warning("⚡ 절반 달성! 계속 실천해보세요!")
        else:
            st.error("🚀 시작이 반입니다! 하나씩 실천해보세요!")

# ---------- 탭4: 제언 및 행동 ----------
with tabs[4]:
    st.header("💡 제언 및 행동")
    st.markdown("데이터를 바탕으로 한 실질적인 개선 방안과 행동 가이드를 제안합니다.")
    
    # 행동 계획 섹션
    action_tabs = st.tabs(["🎯 즉시 실행", "📅 단기 계획", "🏗 장기 비전", "📊 모니터링"])
    
    with action_tabs[0]:
        st.subheader("🎯 오늘부터 할 수 있는 일들")
        
        immediate_actions = {
            "👨‍🎓 개인 차원": {
                "actions": [
                    "쉬는 시간마다 창문 열기 (2-3분)",
                    "마스크 올바르게 착용하기",
                    "실내에서 스프레이 사용 자제",
                    "공기질 앱으로 실시간 확인"
                ],
                "color": COLORS['info']
            },
            "🏫 학급 차원": {
                "actions": [
                    "환기 담당자 정하기 (주별 교대)",
                    "공기청정기 필터 상태 확인",
                    "교실 청소 규칙 재정비",
                    "공기질 측정 기록 시작"
                ],
                "color": COLORS['warning']
            }
        }
        
        cols = st.columns(2)
        for i, (category, data) in enumerate(immediate_actions.items()):
            with cols[i]:
                st.markdown(f"**{category}**")
                for action in data["actions"]:
                    st.write(f"✅ {action}")
    
    with action_tabs[1]:
        st.subheader("📅 1개월 단기 실행 계획")
        
        # 단계별 계획을 시각적으로 표현
        weeks_plan = {
            "1주차": ["환기 습관 형성", "공기질 측정 시작"],
            "2주차": ["데이터 수집 및 분석", "문제점 파악"],
            "3주차": ["개선 방안 실행", "효과 측정"],
            "4주차": ["결과 정리", "확산 계획 수립"]
        }
        
        for week, tasks in weeks_plan.items():
            with st.expander(f"📋 {week} 계획"):
                for task in tasks:
                    st.write(f"• {task}")
    
    with action_tabs[2]:
        st.subheader("🏗 장기 비전 및 정책 제안")
        
        # 정책 제안을 우선순위별로 정리
        policy_suggestions = pd.DataFrame({
            "제안사항": [
                "학교 공기질 측정 의무화",
                "환기 시설 개선 예산 지원",
                "교사 대상 공기질 교육",
                "학부모 참여 모니터링 체계",
                "지역사회 공기질 개선 캠페인"
            ],
            "우선순위": [1, 2, 3, 4, 5],
            "예상기간": ["6개월", "1년", "6개월", "3개월", "지속적"],
            "예상효과": [90, 85, 70, 60, 75]
        })
        
        fig_policy = px.scatter(
            policy_suggestions,
            x="우선순위",
            y="예상효과",
            size="예상효과",
            color="예상기간",
            hover_name="제안사항",
            title="📊 정책 제안사항 우선순위 및 효과 예측",
            labels={"예상효과": "예상 효과 (%)", "우선순위": "우선순위 (낮을수록 우선)"}
        )
        fig_policy.update_layout(height=400)
        st.plotly_chart(fig_policy, use_container_width=True)
        
        # 구체적 실행 방안
        st.markdown("### 🎯 구체적 실행 방안")
        
        execution_plan = {
            "🏫 학교/교육청": [
                "교실별 CO₂ 측정기 설치 (예산: 교실당 15만원)",
                "환기 시설 개선 공사 (예산: 학교당 500만원)", 
                "교사 대상 공기질 관리 연수 프로그램 운영",
                "학교보건법 개정을 통한 공기질 기준 강화"
            ],
            "🏛️ 지방자치단체": [
                "주민참여예산 활용한 공기질 개선 사업",
                "지역 공기질 모니터링 네트워크 구축",
                "시민 대상 실내공기질 교육 프로그램",
                "공공건물 공기질 관리 의무화"
            ],
            "👥 시민사회": [
                "학부모회 주도 공기질 개선 캠페인",
                "청소년 환경 동아리 활동 지원",
                "지역사회 공기질 데이터 공유 플랫폼",
                "전문가-시민 협력 모니터링 체계"
            ]
        }
        
        for category, plans in execution_plan.items():
            with st.expander(f"{category} 실행 방안"):
                for plan in plans:
                    st.write(f"• {plan}")
    
    with action_tabs[3]:
        st.subheader("📊 모니터링 및 평가")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 성과 지표")
            
            # 성과 지표 예시
            kpi_data = pd.DataFrame({
                "지표": ["환기 실행률", "PM2.5 개선율", "건강 증상 감소", "학습 집중도"],
                "목표": [90, 30, 25, 15],
                "현재": [45, 12, 8, 5],
                "달성률": [50, 40, 32, 33]
            })
            
            fig_kpi = px.bar(
                kpi_data,
                x="지표",
                y=["목표", "현재"],
                barmode="group",
                title="📊 주요 성과지표 현황",
                color_discrete_map={"목표": COLORS['success'], "현재": COLORS['warning']}
            )
            fig_kpi.update_layout(height=350)
            st.plotly_chart(fig_kpi, use_container_width=True)
        
        with col2:
            st.markdown("### 📅 점검 일정")
            
            monitoring_schedule = {
                "일일": ["교실 환기 실행 여부", "공기질 측정값 기록"],
                "주간": ["필터 상태 점검", "청소 상태 확인"],
                "월간": ["데이터 분석 및 보고", "개선사항 검토"],
                "분기": ["전체 평가 및 계획 수정", "예산 집행 현황 점검"]
            }
            
            for period, tasks in monitoring_schedule.items():
                st.markdown(f"**{period} 점검**")
                for task in tasks:
                    st.write(f"  ✓ {task}")
                st.write("")
    
    # 행동 다짐서 작성
    st.markdown("---")
    st.subheader("✍️ 나의 공기질 개선 다짐")
    
    with st.form("action_commitment"):
        commitment_text = st.text_area(
            "오늘부터 실천할 구체적인 행동을 작성해주세요:",
            placeholder="예: 매일 쉬는 시간마다 창문을 열어 2분간 환기하겠습니다.",
            height=100
        )
        
        priority_action = st.selectbox(
            "가장 우선적으로 실천할 행동은?",
            ["규칙적 환기", "공기질 측정", "청소 강화", "마스크 착용", "식물 배치", "기타"]
        )
        
        commitment_level = st.slider("실천 의지 수준", 1, 10, 7)
        
        submitted = st.form_submit_button("🎯 다짐 등록")
        
        if submitted:
            if commitment_text:
                st.success("✅ 다짐이 등록되었습니다!")
                st.balloons()
                
                # 다짐 요약 표시
                st.info(f"""
                **나의 다짐:** {commitment_text}
                
                **우선 행동:** {priority_action}
                **의지 수준:** {commitment_level}/10
                
                💪 실천을 통해 더 건강한 환경을 만들어가세요!
                """)
            else:
                st.warning("다짐을 작성해주세요.")
    
    # 추가 리소스
    st.markdown("---")
    st.subheader("📚 참고 자료 및 도움말")
    
    resources_col1, resources_col2 = st.columns(2)
    
    with resources_col1:
        st.markdown("""
        **🔗 유용한 링크**
        - [에어코리아](https://www.airkorea.or.kr/): 실시간 대기질 정보
        - [WHO 실내공기질 가이드라인](https://who.int): 국제 기준
        - [한국환경공단](https://keco.or.kr): 환경 정보 포털
        - [교육부 학교보건포털](https://schoolhealth.kr): 학교 건강 정보
        """)
    
    with resources_col2:
        st.markdown("""
        **📞 문의처**
        - 교육청 시설과: 학교 환경 개선
        - 보건소: 건강 상담
        - 환경청: 대기질 신고
        - 소비자원: 제품 안전성
        """)
    
    # 마무리 메시지
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h3>🌟 함께 만드는 깨끗한 공기</h3>
        <p>작은 실천이 모여 큰 변화를 만듭니다. 오늘부터 시작해보세요!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 최종 다운로드 옵션
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action_plan = f"""
        개인 행동 계획
        
        1. 즉시 실행 사항:
        - 쉬는 시간 환기 (2-3분)
        - 마스크 올바른 착용
        - 공기질 앱 확인
        
        2. 단기 목표 (1개월):
        - 환기 습관 형성
        - 데이터 수집 및 분석
        - 개선 효과 측정
        
        3. 장기 비전:
        - 학급 단위 캠페인
        - 정책 제안 참여
        - 지속적 모니터링
        """
        
        st.download_button(
            "📋 개인 행동계획서 다운로드",
            data=action_plan.encode("utf-8"),
            file_name="personal_action_plan.txt",
            mime="text/plain"
        )
    
    with col2:
        school_proposal = f"""
        학교 대상 개선 제안서
        
        제안 배경:
        - 실내 공기질이 학습 효과에 미치는 영향
        - 청소년 건강 보호의 필요성
        
        구체적 제안:
        1. 교실별 CO₂ 측정기 설치
        2. 환기 시설 개선
        3. 정기적 공기질 점검 체계 구축
        4. 교사 및 학생 교육 프로그램
        
        기대 효과:
        - 집중력 향상 및 학습 효과 증대
        - 호흡기 질환 예방
        - 쾌적한 교육 환경 조성
        """
        
        st.download_button(
            "📄 학교 제안서 다운로드",
            data=school_proposal.encode("utf-8"),
            file_name="school_proposal.txt",
            mime="text/plain"
        )
    
    with col3:
        policy_proposal = f"""
        정책 제안서
        
        현황 및 문제점:
        - 실내공기질 관리 사각지대 존재
        - 측정 및 점검 체계 미비
        - 예산 및 인력 부족
        
        정책 제안:
        1. 실내공기질 관리법 개정
        2. 학교 환경 기준 강화
        3. 예산 지원 확대
        4. 모니터링 체계 구축
        
        추진 방안:
        - 관련 부처 협의
        - 전문가 자문단 구성
        - 시범 사업 실시
        - 단계적 확산
        """
        
        st.download_button(
            "📜 정책 제안서 다운로드",
            data=policy_proposal.encode("utf-8"),
            file_name="policy_proposal.txt",
            mime="text/plain"
        )

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
    <p>⚠️ <strong>주의사항</strong>: 본 대시보드는 교육 및 참고 목적의 예시 데이터와 간단한 모델을 사용합니다.</p>
    <p>실제 의료 상담이나 정책 결정에는 전문 기관의 공식 데이터를 참고하시기 바랍니다.</p>
    <p>📧 문의: air.quality.dashboard@example.com | 📞 상담: 1588-0000</p>
    </div>
    """,
    unsafe_allow_html=True
)

# EOF