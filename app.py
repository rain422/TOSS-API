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
st.markdown("다중 패턴 인식 및 보조지표 필터 융합 시스템")
st.markdown("---")

# --- 2. 사이드바 (컨트롤 패널) ---
with st.sidebar:
    st.header("⚙️ 검색 설정")
    ticker = st.text_input("종목 코드", value="QQQ") 
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
    st.markdown("**📊 AI 보조지표 필터 설정**")
    use_rsi_filter = st.checkbox("RSI 과열 방지 필터 가동", value=True)
    rsi_max = st.slider("RSI 진입 제한 상한선", 50, 80, 65)
    
    use_vol_filter = st.checkbox("거래량 돌파 필터 가동", value=True)
    vol_ratio = st.slider("평균 대비 최소 거래량 (%)", 100, 300, 150, step=10)
    
    st.markdown("---")
    st.markdown("**💰 자본 및 분할 매수 설정**")
    init_cash = st.number_input("총 가상 자본금 ($)", value=10000)
    bet_size = st.number_input("1회 매수 금액 ($)", value=1000)
    target_profit = st.slider("최종 익절 목표 (%)", 0.5, 10.0, 2.0, 0.5)
    stop_loss = st.slider("최종 손절 제한 (%)", 0.5, 10.0, 4.0, 0.5)
    max_adds = st.slider("최대 추가 매수 횟수", 0, 5, 2)
    add_drop_pct = st.slider("추가 매수 하락률 (%)", 1.0, 10.0, 3.0, 0.5)

# --- 3. 데이터 수집 및 보조지표 수학적 연산 함수 ---
@st.cache_data
def get_data_with_indicators(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        return df
    df.reset_index(inplace=True)
    
    # 1차원 플래닝 처리
    close_prices = df['Close'].values.flatten()
    volume_values = df['Volume'].values.flatten()
    
    # [A] RSI 자체 연산 수식
    delta = np.diff(close_prices)
    seed = delta[:14]
    up = seed[seed >= 0].sum() / 14
    down = -seed[seed < 0].sum() / 14
    rs = up / (down + 1e-10)
    rsi = np.zeros_like(close_prices)
    rsi[:14] = 100. - 100. / (1. + rs)
    
    for i in range(14, len(close_prices)):
        d = delta[i-1]
        if d > 0:
            up_val, down_val = d, 0.0
        else:
            up_val, down_val = 0.0, -d
        up = (up * 13 + up_val) / 14
        down = (down * 13 + down_val) / 14
        rs = up / (down + 1e-10)
        rsi[i] = 100. - 100. / (1. + rs)
    df['RSI'] = rsi

    # [B] MACD 자체 연산 수식
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # [C] 5개 캔들 평균 거래량 연산
    df['Vol_Avg5'] = df['Volume'].rolling(window=5).mean()
    
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

# --- 5. 🌟 보조지표 필터가 융합된 백테스팅 엔진 (업그레이드) ---
def run_backtest_with_filters(df, peaks, troughs, tol_percent, tp, sl, time_col, bet, m_adds, drop_pct, 
                              use_rsi, r_max, use_vol, v_ratio):
    highs, lows, closes = df['High'].values.flatten(), df['Low'].values.flatten(), df['Close'].values.flatten()
    volumes, vol_avg5 = df['Volume'].values.flatten(), df['Vol_Avg5'].values.flatten()
    rsi_vals = df['RSI'].values.flatten()
    macd_vals, sig_vals = df['MACD'].values.flatten(), df['SIGNAL'].values.flatten()
    
    trades = []
    in_position, entry_idx, pattern_name = False, 0, ""
    total_shares, total_cost, avg_price, add_count = 0.0, 0.0, 0.0, 0
    
    for i in range(26, len(df)): # MACD 연산을 위해 26번째 캔들부터 시작
        current_close = closes[i]
        
        if not in_position:
            pattern = detect_patterns_at_idx(i, highs, lows, peaks, troughs, tol_percent)
            if pattern:
                # 🌟 다중 센서 필터링 가동
                if use_rsi and rsi_vals[i] > r_max:
                    continue # RSI가 상한선을 넘으면 '가짜 돌파'로 간주하고 진입 차단
                
                if use_vol and not np.isnan(vol_avg5[i]):
                    current_vol_ratio = (volumes[i] / vol_avg5[i]) * 100
                    if current_vol_ratio < v_ratio:
                        continue # 거래량이 실리지 않은 약한 돌파는 진입 차단
                
                # 모든 필터를 통과하면 최종 진입
                in_position, entry_idx, pattern_name = True, i, pattern
                total_shares = bet / current_close
                total_cost = bet
                avg_price = total_cost / total_shares
                add_count = 0
        else:
            ret_from_avg = (current_close - avg_price) / avg_price * 100
            
            if ret_from_avg <= -drop_pct and add_count < m_adds:
                new_shares = bet / current_close
                total_shares += new_shares
                total_cost += bet
                avg_price = total_cost / total_shares
                add_count += 1
                ret_from_avg = (current_close - avg_price) / avg_price * 100
            
            if ret_from_avg >= tp:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name, '투입금액': total_cost, '최종 평단가': avg_price, '청산가': current_close, '수익률(%)': ret_from_avg, '물타기 횟수': add_count, '결과': '익절'})
                in_position = False
            elif ret_from_avg <= -sl:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name, '투입금액': total_cost, '최종 평단가': avg_price, '청산가': current_close, '수익률(%)': ret_from_avg, '물타기 횟수': add_count, '결과': '손절'})
                in_position = False
            elif (i - entry_idx) >= 30:
                trades.append({'진입시간': df.iloc[entry_idx][time_col], '청산시간': df.iloc[i][time_col], '패턴': pattern_name, '투입금액': total_cost, '최종 평단가': avg_price, '청산가': current_close, '수익률(%)': ret_from_avg, '물타기 횟수': add_count, '결과': '타임컷'})
                in_position = False
                
    return pd.DataFrame(trades)

# --- 6. 메인 화면 렌더링 ---
try:
    df = get_data_with_indicators(ticker, period, interval)
    if df.empty or len(df) < 30: st.warning("데이터가 부족합니다.")
    else:
        time_col = df.columns[0]
        highs, lows = df['High'].values.flatten(), df['Low'].values.flatten()
        peaks, _ = find_peaks(highs, distance=distance)
        troughs, _ = find_peaks(-lows, distance=distance)
        
        # 필터가 적용된 백테스팅 가동
        trade_log = run_backtest_with_filters(df, peaks, troughs, tolerance, target_profit, stop_loss, time_col, bet_size, max_adds, add_drop_pct, use_rsi_filter, rsi_max, use_vol_filter, vol_ratio)

        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"${df['Close'].iloc[-1].item():.2f}")
        m2.metric("분석된 캔들", f"{len(df)}개")
        m3.metric("필터링 후 총 매매 횟수", f"{len(trade_log)}회")
        
        tab1, tab2 = st.tabs(["📊 실시간 차트 및 보조지표", "📝 필터링 매매 리포트"])
        
        with tab1:
            fig = go.Figure(data=[go.Candlestick(x=df[time_col], open=df['Open'].values.flatten(), high=highs, low=lows, close=df['Close'].values.flatten(), increasing_line_color='#26a69a', decreasing_line_color='#ef5350')])
            fig.add_trace(go.Scatter(x=df[time_col].iloc[peaks], y=highs[peaks], mode='markers', marker=dict(color='red', size=6), name='저항점'))
            fig.add_trace(go.Scatter(x=df[time_col].iloc[troughs], y=lows[troughs], mode='markers', marker=dict(color='blue', size=6), name='지지점'))
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 
            fig.update_layout(yaxis_title="가격", xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # 아래에 RSI 보조지표 차트 추가로 시각화 효과 극대화
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df[time_col], y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI(14)'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[10, 90]))
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        with tab2:
            if not trade_log.empty:
                trade_log['추정 수익금'] = trade_log['투입금액'] * (trade_log['수익률(%)'] / 100)
                total_profit_sum = trade_log['추정 수익금'].sum()
                win_trades = trade_log[trade_log['결과'] == '익절']
                win_rate = (len(win_trades) / len(trade_log)) * 100
                
                c1, c2 = st.columns(2)
                c1.metric("💵 총 누적 추정 수익금", f"${total_profit_sum:,.2f}")
                c2.metric("🎯 필터링 후 최종 승률", f"{win_rate:.1f}%")
                
                st.markdown("---")
                st.dataframe(trade_log.style.format({
                    '수익률(%)': '{:.2f}%', '최종 평단가': '${:.2f}', '청산가': '${:.2f}',
                    '투입금액': '${:,.2f}', '추정 수익금': '${:,.2f}'
                }), use_container_width=True)
            else: 
                st.info("보조지표 필터 조건이 너무 까다롭거나 패턴 조건에 맞는 매매 내역이 없습니다. 사이드바 설정을 완화해 보세요.")

except Exception as e:
    st.error(f"데이터 처리 중 오류: {e}")
