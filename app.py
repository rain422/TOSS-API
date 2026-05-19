import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# --- 1. 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 트레이딩 대시보드", page_icon="📈", layout="wide")
st.markdown("<style>.block-container { padding-top: 2rem; padding-bottom: 2rem; }</style>", unsafe_allow_html=True)

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
    distance = st.slider("최소 캔들 간격", min_value=3, max_value=20, value=5)
    # 패턴 인식을 위한 허용 오차 (기울기가 0에 얼마나 가까워야 수평으로 인정할지)
    tolerance = st.slider("저항선 수평 허용 오차(%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

# --- 3. 데이터 수집 함수 ---
@st.cache_data
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# --- 4. 패턴 인식 엔진 (수학적 알고리즘) ---
def detect_ascending_triangle(peaks, troughs, highs, lows, tol_percent):
    # 최근 3개의 고점과 저점만 대상으로 분석 (최근 추세 확인)
    if len(peaks) < 3 or len(troughs) < 3:
        return False, None, None
    
    recent_peaks = peaks[-3:]
    recent_troughs = troughs[-3:]
    
    # x값(인덱스)과 y값(가격) 
    peak_x, peak_y = recent_peaks, highs[recent_peaks]
    trough_x, trough_y = recent_troughs, lows[recent_troughs]
    
    # 1차 방정식(y = mx + c) 피팅하여 기울기(m)와 절편(c) 구하기
    peak_slope, peak_intercept = np.polyfit(peak_x, peak_y, 1)
    trough_slope, trough_intercept = np.polyfit(trough_x, trough_y, 1)
    
    # 기울기를 가격 비율로 정규화 (종목마다 가격이 다르므로 %로 변환)
    avg_price = np.mean(highs)
    norm_peak_slope = (peak_slope / avg_price) * 100
    norm_trough_slope = (trough_slope / avg_price) * 100
    
    # 조건 1: 고점 저항선은 수평에 가까운가? (허용 오차 이내)
    is_flat_top = abs(norm_peak_slope) <= tol_percent
    # 조건 2: 저점 지지선은 명확히 상승하고 있는가?
    is_rising_bottom = norm_trough_slope > (tol_percent * 0.5) 
    
    if is_flat_top and is_rising_bottom:
        return True, (peak_slope, peak_intercept, recent_peaks), (trough_slope, trough_intercept, recent_troughs)
    return False, None, None

# --- 5. 메인 화면 레이아웃 및 렌더링 ---
try:
    df = get_data(ticker, period, interval)
    
    if df.empty:
        st.warning("데이터를 불러오지 못했습니다.")
    else:
        # 데이터 및 극점 추출
        highs = df['High'].values.flatten()
        lows = df['Low'].values.flatten()
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)
        
        # 🌟 패턴 인식 엔진 가동
        is_triangle, res_line, sup_line = detect_ascending_triangle(peaks, troughs, highs, lows, tolerance)

        # UI 출력
        current_price = df['Close'].iloc[-1].item()
        prev_price = df['Close'].iloc[-2].item()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="현재가", value=f"${current_price:.2f}", delta=f"{(current_price - prev_price):.2f}")
        m2.metric(label="기간 내 최고가", value=f"${df['High'].max().item():.2f}")
        m3.metric(label="거래량", value=f"{df['Volume'].iloc[-1].item():,}")
        
        tab1, tab2, tab3 = st.tabs(["📊 차트 & 극점 분석", "🤖 패턴 탐지 리포트", "📝 모의투자 (예정)"])
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'].values.flatten(),
                            high=highs, low=lows, close=df['Close'].values.flatten(),
                            increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="Candle")])
            
            fig.add_trace(go.Scatter(x=peaks, y=highs[peaks], mode='markers',
                marker=dict(color='rgba(255, 0, 0, 0.7)', size=8, symbol='triangle-down'), name='저항점'))
            fig.add_trace(go.Scatter(x=troughs, y=lows[troughs], mode='markers',
                marker=dict(color='rgba(0, 150, 255, 0.7)', size=8, symbol='triangle-up'), name='지지점'))

            # 🌟 패턴이 발견되면 차트 위에 추세선(Trendline) 그리기
            if is_triangle:
                p_slope, p_int, p_x = res_line
                t_slope, t_int, t_x = sup_line
                
                # 선을 그릴 시작점과 끝점 설정 (최근 패턴 발생 구간)
                line_x = np.array([min(p_x[0], t_x[0]), df.index[-1]])
                
                fig.add_trace(go.Scatter(x=line_x, y=p_slope * line_x + p_int, 
                                         mode='lines', line=dict(color='yellow', width=2, dash='dash'), name='저항선 (수평)'))
                fig.add_trace(go.Scatter(x=line_x, y=t_slope * line_x + t_int, 
                                         mode='lines', line=dict(color='fuchsia', width=2, dash='dash'), name='지지선 (상승)'))

            fig.update_layout(yaxis_title="가격", xaxis_rangeslider_visible=False, height=600,
                              margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("실시간 패턴 분석 엔진 상태")
            if is_triangle:
                st.success("🚨 **어센딩 트라이앵글(Ascending Triangle) 패턴이 감지되었습니다!**")
                st.write("- 상단 저항선이 수평을 유지하며, 하단 지지선이 상승하며 수렴 중입니다.")
                st.write("- **매매 전략:** 저항선 돌파 시 강한 상승 동력이 발생할 수 있으므로 돌파 매수 타점을 주시하세요.")
            else:
                st.info("현재 식별된 명확한 어센딩 트라이앵글 패턴이 없습니다.")
                st.write("알고리즘이 지속적으로 백그라운드에서 추세선을 연산 중입니다.")

except Exception as e:
    st.error(f"데이터 처리 중 오류: {e}")
