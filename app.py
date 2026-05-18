import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# --- 1. 웹페이지 기본 설정 (항상 최상단) ---
st.set_page_config(page_title="AI 트레이딩 대시보드", page_icon="📈", layout="wide")

# 약간의 CSS를 추가해서 상단 여백을 줄이고 웹앱 느낌을 살림
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# 헤더 영역
st.title("📈 AI 기반 퀀트 트레이딩 대시보드")
st.markdown("단기 추세 분석 및 패턴 인식 모니터링 시스템")
st.markdown("---")

# --- 2. 사이드바 (컨트롤 패널) ---
with st.sidebar:
    st.header("⚙️ 검색 설정")
    ticker = st.text_input("종목 코드 (예: AAPL, QQQ, TSLA)", value="QQQ") 
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1)
    with col2:
        interval = st.selectbox("봉 단위", ["5m", "15m", "1h", "1d"], index=3)
    
    st.markdown("---")
    st.markdown("**🤖 알고리즘 민감도 설정**")
    distance = st.slider("최소 캔들 간격 (노이즈 필터링)", min_value=3, max_value=20, value=5)

# --- 3. 데이터 수집 함수 ---
@st.cache_data
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# --- 4. 메인 화면 레이아웃 및 로직 ---
try:
    df = get_data(ticker, period, interval)
    
    if df.empty:
        st.warning("데이터를 불러오지 못했습니다. 주말이거나 상장 폐지된 종목일 수 있습니다.")
    else:
        # --- [A] 핵심 지표 (Metrics) 전광판 영역 ---
        # 최근 2개의 종가를 가져와서 등락률 계산
        current_price = df['Close'].iloc[-1].item()
        prev_price = df['Close'].iloc[-2].item()
        price_diff = current_price - prev_price
        pct_change = (price_diff / prev_price) * 100
        
        # 3개의 컬럼으로 나누어 지표 표시
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="현재가", value=f"${current_price:.2f}", delta=f"{price_diff:.2f} ({pct_change:.2f}%)")
        m2.metric(label="기간 내 최고가", value=f"${df['High'].max().item():.2f}")
        m3.metric(label="기간 내 최저가", value=f"${df['Low'].min().item():.2f}")
        m4.metric(label="거래량", value=f"{df['Volume'].iloc[-1].item():,}")
        
        st.markdown("<br>", unsafe_allow_html=True) # 줄바꿈

        # --- [B] 탭(Tabs)을 이용한 화면 분할 ---
        tab1, tab2, tab3 = st.tabs(["📊 차트 & 극점 분석", "🤖 패턴 탐지 리포트", "📝 모의투자 시뮬레이션"])
        
        # 데이터를 평탄화(1차원 배열)하여 극점 분석
        highs = df['High'].values.flatten()
        lows = df['Low'].values.flatten()
        
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)

        with tab1:
            # Plotly 차트 생성 (디자인 개선)
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'].values.flatten(),
                            high=highs,
                            low=lows,
                            close=df['Close'].values.flatten(),
                            increasing_line_color='#26a69a', # 트레이딩뷰 스타일 초록
                            decreasing_line_color='#ef5350', # 트레이딩뷰 스타일 빨강
                            name="Candle")])

            # 마커 디자인 개선
            fig.add_trace(go.Scatter(x=peaks, y=highs[peaks], mode='markers',
                marker=dict(color='rgba(255, 0, 0, 0.7)', size=10, symbol='triangle-down'),
                name='저항 (Peaks)'))
            fig.add_trace(go.Scatter(x=troughs, y=lows[troughs], mode='markers',
                marker=dict(color='rgba(0, 150, 255, 0.7)', size=10, symbol='triangle-up'),
                name='지지 (Troughs)'))

            # 차트 배경 및 여백 다듬기
            fig.update_layout(
                yaxis_title="가격 (USD)",
                xaxis_rangeslider_visible=False, # 하단 쓸데없는 슬라이더 제거
                height=650,
                margin=dict(l=0, r=0, t=30, b=0), # 차트 여백 최소화
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified" # 마우스를 올렸을 때 정보가 깔끔하게 나오도록
            )
            # 차트 그리드(격자) 연하게 설정
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("패턴 분석 결과")
            st.info(f"현재 설정된 민감도로 **{len(peaks)}개의 고점**과 **{len(troughs)}개의 저점**이 식별되었습니다.")
            st.write("*(여기에 향후 어센딩 트라이앵글 등 수학적 패턴 분석 로직이 들어갈 예정입니다.)*")
            
        with tab3:
            st.subheader("모의투자 일지 (개발 중)")
            st.write("향후 패턴 탐지 시 자동으로 매수/매도 타점을 기록하고 수익률을 백테스팅하는 공간입니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다. 데이터를 처리하는 중 문제가 생겼습니다: {e}")
