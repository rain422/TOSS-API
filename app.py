import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# --- 1. 웹페이지 기본 설정 ---
st.set_page_config(page_title="패턴 인식 대시보드", layout="wide")
st.title("📈 자동 차트 패턴 인식 프로그램 (v1.0)")

# --- 2. 사이드바 (사용자 입력) ---
with st.sidebar:
    st.header("설정")
    # 예: AAPL(애플), TSLA(테슬라), 005930.KS(삼성전자)
    ticker = st.text_input("종목 코드 (예: AAPL, QQQ)", value="QQQ") 
    period = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("봉 단위", ["5m", "15m", "1h", "1d"], index=3)
    
    st.markdown("---")
    st.markdown("**알고리즘 민감도 설정**")
    # distance: 캔들 몇 개를 간격으로 고점을 찾을지 (노이즈 필터링 역할)
    distance = st.slider("최소 캔들 간격", min_value=3, max_value=20, value=5)

# --- 3. 데이터 수집 함수 ---
@st.cache_data # 데이터를 매번 다시 부르지 않도록 캐싱
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# --- 4. 메인 로직 및 차트 시각화 ---
try:
    # 데이터 불러오기
    df = get_data(ticker, period, interval)
    
    if df.empty:
        st.error("데이터를 불러오지 못했습니다. 종목 코드나 기간을 확인해 주세요.")
    else:
        # SciPy를 이용한 고점(Peaks)과 저점(Troughs) 추출
        # 고가는 그대로, 저가는 음수로 뒤집어서 극소점을 찾음
        highs = df['High'].values.flatten()
        lows = df['Low'].values.flatten()
        
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)

        # Plotly 캔들스틱 차트 생성
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'].values.flatten(),
                        high=highs,
                        low=lows,
                        close=df['Close'].values.flatten(),
                        name="Candle")])

        # 추출된 고점에 빨간색 마커 표시
        fig.add_trace(go.Scatter(
            x=peaks, y=highs[peaks],
            mode='markers',
            marker=dict(color='red', size=8, symbol='triangle-down'),
            name='고점 (Peaks)'
        ))

        # 추출된 저점에 파란색 마커 표시
        fig.add_trace(go.Scatter(
            x=troughs, y=lows[troughs],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='triangle-up'),
            name='저점 (Troughs)'
        ))

        # 차트 레이아웃 다듬기
        fig.update_layout(
            title=f"{ticker} 주가 흐름 및 극점 분석",
            yaxis_title="가격",
            xaxis_title="시간(인덱스)",
            xaxis_rangeslider_visible=False,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"데이터 분석 완료! 총 {len(peaks)}개의 고점과 {len(troughs)}개의 저점을 찾았습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")