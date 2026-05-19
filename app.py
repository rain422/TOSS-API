import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- 1. 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 트레이딩 대시보드", page_icon="📈", layout="wide")
st.markdown("<style>.block-container { padding-top: 2rem; padding-bottom: 2rem; }</style>", unsafe_allow_html=True)

st.title("📈 AI 기반 퀀트 트레이딩 대시보드")
st.markdown("다중 패턴 인식 및 백테스팅 데이터 수집 시스템")
st.markdown("---")

# --- 2. 사이드바 (컨트롤 패널) ---
with st.sidebar:
    st.header("⚙️ 검색 및 매매 설정")
    ticker = st.text_input("종목 코드 (예: AAPL, QQQ, TSLA)", value="QQQ") 
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1)
    with col2:
        interval = st.selectbox("봉 단위", ["5m", "15m", "1h", "1d"], index=1) 
    
    st.markdown("---")
    st.markdown("**🤖 알고리즘 민감도 설정**")
    distance = st.slider("최소 캔들 간격", min_value=3, max_value=20, value=5)
    tolerance = st.slider("수평/대칭 허용 오차(%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

    st.markdown("---")
    st.markdown("**💰 단타 모의매매 규칙**")
    init_cash = st.number_input("초기 자본금 ($)", value=10000)
    target_profit = st.slider("익절 목표 (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    stop_loss = st.slider("손절 제한 (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

# --- 3. 데이터 수집 함수 ---
@st.cache_data
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# --- 4. 다중 패턴 인식 함수 ---
def detect_patterns_at_idx(idx, highs, lows, peaks, troughs, tol_percent):
    avail_peaks = [p for p in peaks if p <= idx]
    avail_troughs = [t for t in troughs if t <= idx]
    
    if len(avail_troughs) >= 2:
        t1, t2 = avail_troughs[-2], avail_troughs[-1]
        if idx - t2 <= 5: 
            p_diff = abs(lows[t1] - lows[t2]) / lows[t1] * 100
            if p_diff <= (tol_percent * 2):
                return "쌍바닥"

    if len(avail_peaks) >= 3 and len(avail_troughs) >= 3:
        p_x, p_y = avail_peaks[-3:], highs[avail_peaks[-3:]]
        t_x, t_y = avail_troughs[-3:], lows[avail_troughs[-3:]]
        
        p_slope, _ = np.polyfit(p_x, p_y, 1)
        t_slope, _ = np.polyfit(t_x, t_y, 1)
        
        avg_price = np.mean(highs[:idx+1])
        norm_p_slope = (p_slope / avg_price) * 100
        norm_t_slope = (t_slope / avg_price) * 100
        
        is_flat_top = abs(norm_p_slope) <= tol_percent
        is_flat_bottom = abs(norm_t_slope) <= tol_percent
        is_rising_bottom = norm_t_slope > tol_percent
        is_falling_top = norm_p_slope < -tol_percent
        
        if is_flat_top and is_rising_bottom: return "어센딩 트라이앵글"
        elif is_falling_top and is_flat_bottom: return "디센딩 트라이앵글"
        elif is_falling_top and is_rising_bottom: return "대칭 삼각수렴"
        elif is_flat_top and is_flat_bottom: return "박스권 채널"
        
    return None

# --- 5. 백테스팅 매매 시뮬레이터 엔진 ---
def run_backtest(df, peaks, troughs, tol_percent, tp, sl):
    highs = df['High'].values.flatten()
    lows = df['Low'].values.flatten()
    closes = df['Close'].values.flatten()
    
    # 🌟 에러 원인 해결: 자동으로 날짜/시간 컬럼 이름 찾기
    time_col = df.columns[0] 
    
    trades = []
    in_position = False
    entry_price = 0
    entry_idx = 0
    pattern_name = ""
    
    for i in range(20, len(df)):
        if not in_position:
            pattern = detect_patterns_at_idx(i, highs, lows, peaks, troughs, tol_percent)
            if pattern:
                in_position = True
                entry_price = closes[i]
                entry_idx = i
                pattern_name = pattern
        else:
            current_close = closes[i]
            ret = (current_close - entry_price) / entry_price * 100
            
            # 🌟 에러 원인 해결: 찾은 time_col을 이용해 정확한 시간 기록
            if ret >= tp:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], 
                               '패턴': pattern_name, '진입가': entry_price, '청산가': current_close, '수익률(%)': ret, '결과': '익절'})
                in_position = False
            elif ret <= -sl:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], 
                               '패턴': pattern_name, '진입가': entry_price, '청산가': current_close, '수익률(%)': ret, '결과': '손절'})
                in_position = False
            elif (i - entry_idx) >= 15:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], 
                               '패턴': pattern_name, '진입가': entry_price, '청산가': current_close, '수익률(%)': ret, '결과': '타임컷'})
                in_position = False
                
    return pd.DataFrame(trades)

# --- 6. 메인 화면 레이아웃 및 렌더링 ---
try:
    df = get_data(ticker, period, interval)
    
    # 데이터가 너무 짧아서 분석할 수 없는 경우 예외 처리
    if df.empty or len(df) < 30:
        st.warning("데이터를 불러오지 못했거나 분석하기에 캔들 개수가 너무 적습니다. 기간을 늘려주세요.")
    else:
        time_col = df.columns[0] # 차트 x축용 시간 데이터 추출
        
        highs = df['High'].values.flatten()
        lows = df['Low'].values.flatten()
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)
        
        trade_log = run_backtest(df, peaks, troughs, tolerance, target_profit, stop_loss)

        current_price = df['Close'].iloc[-1].item()
        prev_price = df['Close'].iloc[-2].item()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(label="현재가", value=f"${current_price:.2f}", delta=f"{(current_price - prev_price):.2f}")
        m2.metric(label="기간 내 최고가", value=f"${df['High'].max().item():.2f}")
        m3.metric(label="분석된 총 캔들 개수", value=f"{len(df)}개")
        m4.metric(label="검출된 총 매매 횟수", value=f"{len(trade_log)}회")
        
        tab1, tab2 = st.tabs(["📊 실시간 차트 분석", "📝 모의투자 백테스팅 리포트"])
        
        with tab1:
            # 🌟 에러 원인 해결: x축에 정확한 날짜 데이터(time_col) 삽입
            fig = go.Figure(data=[go.Candlestick(x=df[time_col], open=df['Open'].values.flatten(),
                            high=highs, low=lows, close=df['Close'].values.flatten(),
                            increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="Candle")])
            
            fig.add_trace(go.Scatter(x=df[time_col].iloc[peaks], y=highs[peaks], mode='markers', marker=dict(color='red', size=6), name='저항점'))
            fig.add_trace(go.Scatter(x=df[time_col].iloc[troughs], y=lows[troughs], mode='markers', marker=dict(color='blue', size=6), name='지지점'))
            
            # 주말/야간 빈 공간 숨기기 처리
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
            fig.update_layout(yaxis_title="가격", xaxis_rangeslider_visible=False, height=600, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("📊 과거 데이터 기반 매매 시뮬레이션 결과")
            
            if trade_log.empty:
                st.info("설정된 기간 및 민감도 내에서 매매 조건(패턴 돌파 후 익청/손절)을 만족한 거래 내역이 없습니다.")
            else:
                win_trades = trade_log[trade_log['결과'] == '익절']
                win_rate = (len(win_trades) / len(trade_log)) * 100
                total_return = trade_log['수익률(%)'].sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric(label="📈 총 누적 수익률", value=f"{total_return:.2f}%")
                c2.metric(label="🎯 매매 승률", value=f"{win_rate:.1f}%")
                c3.metric(label="💰 추정 최종 자산", value=f"${init_cash * (1 + total_return/100):,.2f}")
                
                st.markdown("---")
                st.subheader("📝 상세 매매 일지 (AI 학습용 원천 데이터)")
                # 시간 데이터를 보기 좋게 포맷팅
                st.dataframe(trade_log.style.format({'수익률(%)': '{:.2f}%', '진입가': '${:.2f}', '청산가': '${:.2f}'}), use_container_width=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류: {e}")
