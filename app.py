import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import itertools # 조합 계산을 위한 라이브러리

# --- 1. 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 트레이딩 대시보드", page_icon="📈", layout="wide")
st.markdown("<style>.block-container { padding-top: 2rem; padding-bottom: 2rem; }</style>", unsafe_allow_html=True)

st.title("📈 AI 기반 퀀트 트레이딩 대시보드")
st.markdown("다중 패턴 인식 및 AI 파라미터 최적화 시스템")
st.markdown("---")

# --- 2. 사이드바 (컨트롤 패널) ---
with st.sidebar:
    st.header("⚙️ 검색 설정")
    ticker = st.text_input("종목 코드 (예: AAPL, QQQ, TSLA)", value="QQQ") 
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1)
    with col2:
        interval = st.selectbox("봉 단위", ["5m", "15m", "1h", "1d"], index=1) 
    
    st.markdown("---")
    st.markdown("**🛠️ 수동 테스트 설정**")
    distance = st.slider("최소 캔들 간격", 3, 20, 5)
    tolerance = st.slider("수평/대칭 허용 오차(%)", 0.0, 2.0, 0.5, 0.1)
    init_cash = st.number_input("초기 자본금 ($)", value=10000)
    target_profit = st.slider("익절 목표 (%)", 0.5, 5.0, 1.5, 0.1)
    stop_loss = st.slider("손절 제한 (%)", 0.5, 5.0, 1.0, 0.1)

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
            if p_diff <= (tol_percent * 2): return "쌍바닥"

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

# --- 5. 백테스팅 매매 엔진 ---
def run_backtest(df, peaks, troughs, tol_percent, tp, sl, time_col):
    highs, lows, closes = df['High'].values.flatten(), df['Low'].values.flatten(), df['Close'].values.flatten()
    trades = []
    in_position, entry_price, entry_idx, pattern_name = False, 0, 0, ""
    
    for i in range(20, len(df)):
        if not in_position:
            pattern = detect_patterns_at_idx(i, highs, lows, peaks, troughs, tol_percent)
            if pattern:
                in_position, entry_price, entry_idx, pattern_name = True, closes[i], i, pattern
        else:
            current_close = closes[i]
            ret = (current_close - entry_price) / entry_price * 100
            
            if ret >= tp:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name, '진입가': entry_price, '청산가': current_close, '수익률(%)': ret, '결과': '익절'})
                in_position = False
            elif ret <= -sl:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name, '진입가': entry_price, '청산가': current_close, '수익률(%)': ret, '결과': '손절'})
                in_position = False
            elif (i - entry_idx) >= 15:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name, '진입가': entry_price, '청산가': current_close, '수익률(%)': ret, '결과': '타임컷'})
                in_position = False
                
    return pd.DataFrame(trades)

# --- 6. 🌟 AI 최적화 알고리즘 엔진 ---
def optimize_strategy(df, peaks, troughs, time_col):
    # 테스트할 변수들의 범위 (Grid)
    tp_range = [1.0, 1.5, 2.0, 3.0] # 익절 %
    sl_range = [0.5, 1.0, 1.5, 2.0] # 손절 %
    tol_range = [0.2, 0.5, 0.8]     # 오차율 %
    
    best_return = -999
    best_params = {}
    best_log = pd.DataFrame()
    
    # 모든 경우의 수를 순회하며 시뮬레이션
    combinations = list(itertools.product(tp_range, sl_range, tol_range))
    
    for tp, sl, tol in combinations:
        log = run_backtest(df, peaks, troughs, tol, tp, sl, time_col)
        if not log.empty:
            total_ret = log['수익률(%)'].sum()
            if total_ret > best_return:
                best_return = total_ret
                best_params = {'익절': tp, '손절': sl, '오차율': tol}
                best_log = log
                
    return best_params, best_return, best_log

# --- 7. 메인 화면 렌더링 ---
try:
    df = get_data(ticker, period, interval)
    if df.empty or len(df) < 30: st.warning("데이터가 부족합니다.")
    else:
        time_col = df.columns[0]
        highs, lows = df['High'].values.flatten(), df['Low'].values.flatten()
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)
        
        # 기본 수동 백테스팅
        trade_log = run_backtest(df, peaks, troughs, tolerance, target_profit, stop_loss, time_col)

        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"${df['Close'].iloc[-1].item():.2f}")
        m2.metric("분석된 캔들", f"{len(df)}개")
        m3.metric("수동 매매 횟수", f"{len(trade_log)}회")
        
        tab1, tab2, tab3 = st.tabs(["📊 실시간 차트", "📝 수동 백테스팅 리포트", "🤖 AI 자동 최적화 (Tuning)"])
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=df[time_col], open=df['Open'].values.flatten(), high=highs, low=lows, close=df['Close'].values.flatten(), increasing_line_color='#26a69a', decreasing_line_color='#ef5350')])
            fig.add_trace(go.Scatter(x=df[time_col].iloc[peaks], y=highs[peaks], mode='markers', marker=dict(color='red', size=6), name='저항점'))
            fig.add_trace(go.Scatter(x=df[time_col].iloc[troughs], y=lows[troughs], mode='markers', marker=dict(color='blue', size=6), name='지지점'))
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
            fig.update_layout(yaxis_title="가격", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            if not trade_log.empty:
                st.metric("📈 총 누적 수익률", f"{trade_log['수익률(%)'].sum():.2f}%")
                st.dataframe(trade_log.style.format({'수익률(%)': '{:.2f}%'}), use_container_width=True)
            else: st.info("수동 조건에 맞는 매매 내역이 없습니다.")
                
        # 🌟 새로운 알고리즘 탭
        with tab3:
            st.subheader("🚀 알고리즘 기반 황금 세팅 찾기")
            st.write("프로그램이 백그라운드에서 수십 가지의 익절/손절/오차율 조합을 돌려보고, 가장 높은 누적 수익률을 내는 최적의 값을 찾아냅니다.")
            
            if st.button("🔥 자동 최적화 시작", use_container_width=True):
                with st.spinner("알고리즘이 수십 번의 시뮬레이션을 돌리며 최적값을 계산 중입니다... (약 5~10초 소요)"):
                    best_params, best_ret, best_log = optimize_strategy(df, peaks, troughs, time_col)
                
                if best_params:
                    st.success("🎉 최적화 완료! 이 종목의 최고 효율 세팅값을 찾았습니다.")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🏆 최대 누적 수익률", f"{best_ret:.2f}%")
                    c2.metric("최적 익절 라인", f"{best_params['익절']}%")
                    c3.metric("최적 손절 라인", f"{best_params['손절']}%")
                    c4.metric("최적 오차율", f"{best_params['오차율']}%")
                    
                    st.markdown("---")
                    st.write("해당 세팅으로 진행된 시뮬레이션 매매 일지")
                    st.dataframe(best_log.style.format({'수익률(%)': '{:.2f}%'}), use_container_width=True)
                else:
                    st.warning("어떤 조합으로도 수익이 나는 타점을 찾지 못했습니다.")

except Exception as e:
    st.error(f"데이터 처리 중 오류: {e}")
