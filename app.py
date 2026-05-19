import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import itertools

# --- 1. 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 트레이딩 대시보드", page_icon="📈", layout="wide")
st.markdown("<style>.block-container { padding-top: 2rem; padding-bottom: 2rem; }</style>", unsafe_allow_html=True)

st.title("📈 AI 기반 퀀트 트레이딩 대시보드")
st.markdown("다중 패턴 인식 및 분할 매수(물타기) 시뮬레이션 시스템")
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
    st.markdown("**🤖 패턴 민감도 설정**")
    distance = st.slider("최소 캔들 간격", 3, 20, 5)
    tolerance = st.slider("수평/대칭 허용 오차(%)", 0.0, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("**💰 기본 매매 규칙**")
    init_cash = st.number_input("총 가상 자본금 ($)", value=10000)
    bet_size = st.number_input("1회 매수 투입 금액 ($)", value=1000)
    target_profit = st.slider("최종 익절 목표 (%)", 0.5, 10.0, 2.0, 0.5)
    stop_loss = st.slider("최종 손절 제한 (%)", 0.5, 10.0, 5.0, 0.5)
    
    st.markdown("---")
    st.markdown("**📉 분할 매수 (물타기) 설정**")
    max_adds = st.slider("최대 추가 매수 횟수", 0, 5, 2)
    add_drop_pct = st.slider("추가 매수 발동 하락률 (%)", 1.0, 10.0, 3.0, 0.5)

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

# --- 5. 🌟 평균단가 계산 및 백테스팅 엔진 (업그레이드) ---
def run_backtest(df, peaks, troughs, tol_percent, tp, sl, time_col, bet, m_adds, drop_pct):
    highs, lows, closes = df['High'].values.flatten(), df['Low'].values.flatten(), df['Close'].values.flatten()
    trades = []
    
    in_position = False
    entry_idx = 0
    pattern_name = ""
    
    # 계좌 및 평균단가 관리 변수
    total_shares = 0.0
    total_cost = 0.0
    avg_price = 0.0
    add_count = 0
    
    for i in range(20, len(df)):
        current_close = closes[i]
        
        if not in_position:
            pattern = detect_patterns_at_idx(i, highs, lows, peaks, troughs, tol_percent)
            if pattern:
                # 첫 진입 (1차 매수)
                in_position = True
                entry_idx = i
                pattern_name = pattern
                
                total_shares = bet / current_close
                total_cost = bet
                avg_price = total_cost / total_shares
                add_count = 0
        else:
            # 포지션 유지 중: 현재 평균단가 대비 수익률 계산
            ret_from_avg = (current_close - avg_price) / avg_price * 100
            
            # [A] 추가 매수 (물타기) 로직
            if ret_from_avg <= -drop_pct and add_count < m_adds:
                new_shares = bet / current_close
                total_shares += new_shares
                total_cost += bet
                
                # 새로운 평균단가 재계산 수식: (기존 투입액 + 신규 투입액) / (기존 수량 + 신규 수량)
                avg_price = total_cost / total_shares
                add_count += 1
                
                # 추가 매수 후 새롭게 바뀐 수익률로 다시 갱신
                ret_from_avg = (current_close - avg_price) / avg_price * 100
            
            # [B] 청산 로직 (바뀐 평균단가 기준)
            if ret_from_avg >= tp:
                trades.append({
                    '진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name,
                    '투입금액': total_cost, '최종 평단가': avg_price, '청산가': current_close,
                    '수익률(%)': ret_from_avg, '물타기 횟수': add_count, '결과': '익절'
                })
                in_position = False
            elif ret_from_avg <= -sl:
                trades.append({
                    '진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name,
                    '투입금액': total_cost, '최종 평단가': avg_price, '청산가': current_close,
                    '수익률(%)': ret_from_avg, '물타기 횟수': add_count, '결과': '손절'
                })
                in_position = False
            elif (i - entry_idx) >= 30: # 단타 타임컷 기한 연장 (분할 매수를 고려)
                trades.append({
                    '진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name,
                    '투입금액': total_cost, '최종 평단가': avg_price, '청산가': current_close,
                    '수익률(%)': ret_from_avg, '물타기 횟수': add_count, '결과': '타임컷'
                })
                in_position = False
                
    return pd.DataFrame(trades)

# --- 6. AI 최적화 알고리즘 엔진 (물타기 변수 추가) ---
def optimize_strategy(df, peaks, troughs, time_col, bet, m_adds, drop_pct):
    tp_range = [1.5, 3.0, 5.0] 
    sl_range = [2.0, 4.0, 6.0] 
    tol_range = [0.2, 0.5, 0.8]     
    
    best_return = -999
    best_params = {}
    best_log = pd.DataFrame()
    
    combinations = list(itertools.product(tp_range, sl_range, tol_range))
    
    for tp, sl, tol in combinations:
        log = run_backtest(df, peaks, troughs, tol, tp, sl, time_col, bet, m_adds, drop_pct)
        if not log.empty:
            # 수익률(%)이 아니라 투입 금액 대비 실제 벌어들인 '수익금' 기반으로 최적화
            log['수익금'] = log['투입금액'] * (log['수익률(%)'] / 100)
            total_profit = log['수익금'].sum()
            
            if total_profit > best_return:
                best_return = total_profit
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
        
        # 🌟 업그레이드된 백테스팅 실행
        trade_log = run_backtest(df, peaks, troughs, tolerance, target_profit, stop_loss, time_col, bet_size, max_adds, add_drop_pct)

        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"${df['Close'].iloc[-1].item():.2f}")
        m2.metric("분석된 캔들", f"{len(df)}개")
        m3.metric("수동 매매 횟수", f"{len(trade_log)}회")
        
        tab1, tab2, tab3 = st.tabs(["📊 실시간 차트", "📝 물타기 백테스팅 리포트", "🤖 AI 자동 최적화"])
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=df[time_col], open=df['Open'].values.flatten(), high=highs, low=lows, close=df['Close'].values.flatten(), increasing_line_color='#26a69a', decreasing_line_color='#ef5350')])
            fig.add_trace(go.Scatter(x=df[time_col].iloc[peaks], y=highs[peaks], mode='markers', marker=dict(color='red', size=6), name='저항점'))
            fig.add_trace(go.Scatter(x=df[time_col].iloc[troughs], y=lows[troughs], mode='markers', marker=dict(color='blue', size=6), name='지지점'))
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
            fig.update_layout(yaxis_title="가격", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            if not trade_log.empty:
                # 수익금 계산 컬럼 추가
                trade_log['추정 수익금'] = trade_log['투입금액'] * (trade_log['수익률(%)'] / 100)
                total_profit_sum = trade_log['추정 수익금'].sum()
                
                st.metric("💵 총 누적 추정 수익금", f"${total_profit_sum:,.2f}")
                st.dataframe(trade_log.style.format({
                    '수익률(%)': '{:.2f}%', '최종 평단가': '${:.2f}', '청산가': '${:.2f}',
                    '투입금액': '${:,.2f}', '추정 수익금': '${:,.2f}'
                }), use_container_width=True)
            else: st.info("조건에 맞는 매매 내역이 없습니다.")
                
        with tab3:
            st.subheader("🚀 알고리즘 기반 황금 세팅 찾기")
            if st.button("🔥 물타기 전략 포함 자동 최적화 시작", use_container_width=True):
                with st.spinner("알고리즘이 수십 번의 시뮬레이션을 돌리며 최적값을 계산 중입니다..."):
                    best_params, best_ret, best_log = optimize_strategy(df, peaks, troughs, time_col, bet_size, max_adds, add_drop_pct)
                
                if best_params:
                    st.success("🎉 최적화 완료!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🏆 최대 누적 수익금", f"${best_ret:,.2f}")
                    c2.metric("최적 익절 라인", f"{best_params['익절']}%")
                    c3.metric("최적 손절 라인", f"{best_params['손절']}%")
                    c4.metric("최적 오차율", f"{best_params['오차율']}%")
                    
                    best_log['추정 수익금'] = best_log['투입금액'] * (best_log['수익률(%)'] / 100)
                    st.dataframe(best_log.style.format({
                        '수익률(%)': '{:.2f}%', '최종 평단가': '${:.2f}', '청산가': '${:.2f}',
                        '투입금액': '${:,.2f}', '추정 수익금': '${:,.2f}'
                    }), use_container_width=True)
                else:
                    st.warning("수익이 나는 타점을 찾지 못했습니다.")

except Exception as e:
    st.error(f"데이터 처리 중 오류: {e}")
