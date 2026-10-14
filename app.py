# ==============================================================================
# STREAMLIT QUANT SCANNER WEB APP PRO (INSTITUTIONAL GRADE v2.0)
# Donchian Volatility Breakout + Triple Barrier + Meta-Labeling ML + 
# Market Regime Filter (SPY > SMA200) + Dynamic Half-Kelly Sizing + Telegram
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Configuración de la página Web
st.set_page_config(
    page_title="Master Quant Scanner Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado Dark Mode Neón
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    .metric-card { background-color: #1e293b; padding: 15px; border-radius: 10px; border-left: 4px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PANEL LATERAL (SIDEBAR DE CONFIGURACIÓN)
# ------------------------------------------------------------------------------
st.sidebar.title("🎛️ Configuración Avanzada")
capital = st.sidebar.number_input("Capital Total ($ USD)", value=100000.0, step=5000.0)
riesgo_base_pct = st.sidebar.slider("Riesgo Base por Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
prob_threshold = st.sidebar.slider("Umbral de Seguridad ML", min_value=0.40, max_value=0.70, value=0.50, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Gestión Institucional de Riesgo")
activar_filtro_regimen = st.sidebar.checkbox("Filtro de Régimen de Mercado (SPY > SMA200)", value=True)
usar_kelly_dinamico = st.sidebar.checkbox("Dimensionamiento Dinámico Half-Kelly", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📱 Alertas Telegram")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")

# ------------------------------------------------------------------------------
# INTERFAZ PRINCIPAL CON WATCHLIST FIJA
# ------------------------------------------------------------------------------
st.title("⚡ Master Quantitative Market Scanner Pro")
st.caption("Donchian Volatility Breakout + Triple Barrier + Meta-Labeling ML + Régimen Macro SPY + Kelly Fraccionado")

# Watchlist Fija personalizada
tickers_input = st.text_area(
    "Watchlist de Activos Monitoreados (separados por coma):",
    value="SPY, QQQ, NVDA, AAPL, MSFT, AMZN, TSLA, META, BTC-USD, MELI, OKLO, QCOM, AVGO, YPF, TGS, MU, NU, ASML, IREN, VIST, VT, PAMP, IBIT, TSM, SPCX, BRK-B",
    height=100
)

# ------------------------------------------------------------------------------
# FILTRO DE RÉGIMEN DE MERCADO (MACRO FILTER)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def obtener_regimen_mercado():
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)
        if spy.empty or len(spy) < 200:
            return {"alcista": True, "precio": 0, "sma200": 0, "msg": "Datos SPY insuficientes"}
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.xs("SPY", axis=1, level=1) if "SPY" in spy.columns.levels[1] else spy.droplevel(1, axis=1)
        
        spy['SMA200'] = spy['Close'].rolling(200).mean()
        last_row = spy.iloc[-1]
        precio_spy = float(last_row['Close'])
        sma200_spy = float(last_row['SMA200'])
        
        es_alcista = precio_spy >= sma200_spy
        return {
            "alcista": es_alcista,
            "precio": round(precio_spy, 2),
            "sma200": round(sma200_spy, 2),
            "msg": f"SPY ${precio_spy:.2f} vs SMA200${sma200_spy:.2f}"
        }
    except Exception as e:
        return {"alcista": True, "precio": 0, "sma200": 0, "msg": f"Error descargando SPY: {e}"}

# ------------------------------------------------------------------------------
# MOTOR CUANTITATIVO
# ------------------------------------------------------------------------------
def procesar_activo(ticker, capital_val, riesgo_base_val, threshold_val, regime_ok, kelly_enabled):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty or len(df) < 100:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df.droplevel(1, axis=1)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        entry_window, exit_window = 20, 10
        pt_mult, sl_mult, num_days = 1.8, 1.2, 5

        # ATR(14)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        df['Donchian_High'] = df['High'].shift(1).rolling(entry_window).max()
        df['Donchian_Low'] = df['Low'].shift(1).rolling(exit_window).min()

        sig = np.where(df['Close'] > df['Donchian_High'], 1, np.where(df['Close'] < df['Donchian_Low'], 0, np.nan))
        df['Primary_Signal'] = pd.Series(sig, index=df.index).ffill().fillna(0)
        df['Primary_Entry'] = np.where((df['Primary_Signal'] == 1) & (df['Primary_Signal'].shift(1) == 0), 1, 0)

        # Triple Barrier Method
        labels = []
        indices = df.index
        entries = df[df['Primary_Entry'] == 1].index

        for entry_time in entries:
            loc = indices.get_loc(entry_time)
            if loc + num_days >= len(df):
                continue

            entry_price = df.loc[entry_time, 'Close']
            atr = df.loc[entry_time, 'ATR']

            target_pt = entry_price + (pt_mult * atr)
            target_sl = entry_price - (sl_mult * atr)

            future_prices = df.iloc[loc+1 : loc+1+num_days]
            label = 0
            for t, row in future_prices.iterrows():
                if row['High'] >= target_pt:
                    label = 1
                    break
                elif row['Low'] <= target_sl:
                    label = 0
                    break

            labels.append((entry_time, label))

        labels_df = pd.DataFrame(labels, columns=['Date', 'Meta_Label']).set_index('Date')
        df = df.join(labels_df, how='left')

        # Features ML
        df['Vol_20d'] = df['Close'].pct_change().rolling(20).std()
        df['ATR_Norm'] = df['ATR'] / df['Close']
        df['Dist_Donchian_High'] = (df['Close'] - df['Donchian_High']) / df['ATR']
        df['Vol_Ratio'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Return_5d'] = df['Close'].pct_change(5)

        df_clean = df.dropna()
        data_labeled = df_clean[df_clean['Primary_Entry'] == 1].dropna(subset=['Meta_Label'])
        feature_cols = ['Vol_20d', 'ATR_Norm', 'Dist_Donchian_High', 'Vol_Ratio', 'RSI', 'Return_5d']

        if len(data_labeled) < 8:
            ml_prob = 0.50
        else:
            X = data_labeled[feature_cols]
            y = data_labeled['Meta_Label']
            clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            clf.fit(X, y)
            last_features = df_clean[feature_cols].iloc[[-1]]
            ml_prob = float(clf.predict_proba(last_features)[0][1])

        last_row = df_clean.iloc[-1]
        precio_cierre = float(last_row['Close'])
        atr_actual = float(last_row['ATR'])
        primary_signal = int(last_row['Primary_Signal'])
        primary_entry = int(last_row['Primary_Entry'])

        distancia_sl_pts = atr_actual * sl_mult
        precio_sl = precio_cierre - distancia_sl_pts
        precio_tp = precio_cierre + (atr_actual * pt_mult)

        # Dimensionamiento Dinámico (Half-Kelly + ATR Sizing)
        b_payoff = pt_mult / sl_mult
        p = ml_prob
        
        f_kelly = (p * b_payoff - (1 - p)) / b_payoff if b_payoff > 0 else 0
        half_kelly_frac = max(0.0, 0.5 * f_kelly)

        if kelly_enabled and p >= threshold_val:
            factor_escalado = half_kelly_frac / 0.125
            factor_escalado = max(0.5, min(2.5, factor_escalado))
            riesgo_efectivo_pct = riesgo_base_val * factor_escalado
        else:
            riesgo_efectivo_pct = riesgo_base_val

        riesgo_usd = capital_val * (riesgo_efectivo_pct / 100.0)
        unidades = math.floor(riesgo_usd / distancia_sl_pts) if distancia_sl_pts > 0 else 0
        monto_total_usd = unidades * precio_cierre

        if not regime_ok and activar_filtro_regimen and primary_signal == 1:
            estado_str = "⛔ BLOQUEADO (MERCADO BAJISTA)"
            prioridad = 4
        elif primary_signal == 1 and ml_prob >= threshold_val:
            estado_str = "🟢 COMPRA NUEVA" if primary_entry == 1 else "🔵 MANTENER"
            prioridad = 1
        elif primary_signal == 1 and ml_prob < threshold_val:
            estado_str = "🟡 RECHAZADA ML"
            prioridad = 2
        else:
            estado_str = "⚪ LIQUIDEZ"
            prioridad = 3

        return {
            "Ticker": ticker.upper(),
            "Precio": round(precio_cierre, 2),
            "Estado": estado_str,
            "Confianza ML": f"{ml_prob*100:.1f}%",
            "Half-Kelly %": f"{half_kelly_frac*100:.1f}%",
            "Riesgo Usado": f"{riesgo_efectivo_pct:.2f}%",
            "Stop Loss ($)": round(precio_sl, 2) if primary_signal == 1 else 0.0,
            "Take Profit ($)": round(precio_tp, 2) if primary_signal == 1 else 0.0,
            "Unidades": unidades if prioridad == 1 else 0,
            "Capital ($)": round(monto_total_usd, 2) if prioridad == 1 else 0.0,
            "_df": df_clean.tail(120),
            "_prioridad": prioridad,
            "_prob_raw": ml_prob
        }
    except Exception:
        return None

# Función para generar gráficos con Plotly
def crear_grafico_donchian(ticker, df_hist, precio_sl, precio_tp, estado):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hist.index, y=df_hist['Close'],
        mode='lines', name='Precio Cierre',
        line=dict(color='#38bdf8', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_hist.index, y=df_hist['Donchian_High'],
        mode='lines', name='Donchian High (20d)',
        line=dict(color='#22c55e', width=1.5, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df_hist.index, y=df_hist['Donchian_Low'],
        mode='lines', name='Donchian Low (10d)',
        line=dict(color='#ef4444', width=1.5, dash='dash')
    ))

    if "COMPRA" in estado or "MANTENER" in estado:
        last_date = df_hist.index[-1]
        fig.add_trace(go.Scatter(
            x=[last_date], y=[precio_tp],
            mode='markers+text', name='Take Profit',
            marker=dict(color='#22c55e', size=10, symbol='triangle-up'),
            text=[f"TP: ${precio_tp:,.2f}"], textposition="top center"
        ))
        fig.add_trace(go.Scatter(
            x=[last_date], y=[precio_sl],
            mode='markers+text', name='Stop Loss',
            marker=dict(color='#ef4444', size=10, symbol='triangle-down'),
            text=[f"SL: ${precio_sl:,.2f}"], textposition="bottom center"
        ))

    fig.update_layout(
        title=f"Evolución Técnica y Canales de Donchian — {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio ($ USD)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# ------------------------------------------------------------------------------
# VERIFICACIÓN DE RÉGIMEN DE MERCADO
# ------------------------------------------------------------------------------
regimen_info = obtener_regimen_mercado()
if activar_filtro_regimen:
    if regimen_info['alcista']:
        st.success(f"🌐 **Régimen Macro Favorables (BULLISH):** {regimen_info['msg']}. Mercado apto para posiciones de Breakout.")
    else:
        st.warning(f"⚠️ **Régimen Macro Cauteloso (BEARISH):** {regimen_info['msg']}. Las entradas en largo se bloquearán automáticamente para proteger el portafolio.")

# Botón de ejecución
if st.button("🚀 Ejecutar Escáner Cuantitativo Avanzado", type="primary", use_container_width=True):
    lista_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not lista_tickers:
        st.error("Ingresá al menos un ticker válido.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        resultados = []

        regime_ok = regimen_info['alcista']

        for i, tk in enumerate(lista_tickers):
            status_text.text(f"Analizando {tk} ({i+1}/{len(lista_tickers)})...")
            res = procesar_activo(tk, capital, riesgo_base_pct, prob_threshold, regime_ok, usar_kelly_dinamico)
            if res:
                resultados.append(res)
            progress_bar.progress((i + 1) / len(lista_tickers))

        status_text.empty()
        progress_bar.empty()

        if resultados:
            df_res = pd.DataFrame(resultados).sort_values(by=["_prioridad", "_prob_raw"], ascending=[True, False])
            df_activas = df_res[df_res['_prioridad'] == 1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Activos Escaneados", len(lista_tickers))
            col2.metric("Oportunidades Aprobadas", len(df_activas))
            col3.metric("Capital Total Asignado", f"${df_activas['Capital ($)'].sum():,.2f} USD")

            st.markdown("---")
            st.markdown("### 🎯 Oportunidades Activas Aprobadas (con Kelly Dinámico)")
            if not df_activas.empty:
                st.dataframe(
                    df_activas[['Ticker', 'Precio', 'Estado', 'Confianza ML', 'Half-Kelly %', 'Riesgo Usado', 'Stop Loss ($)', 'Take Profit ($)', 'Unidades', 'Capital ($)']],
                    use_container_width=True
                )
            else:
                st.info("No hay señales de compra que superen los filtros de Régimen Macro y ML en este momento.")

            st.markdown("---")
            st.markdown("### 📊 Gráficos Interactivos de Activos")
            
            opciones_tickers = [r['Ticker'] for r in resultados]
            ticker_seleccionado = st.selectbox("🔍 Elegí el activo que querés analizar en el gráfico:", opciones_tickers)
            
            res_sel = next((r for r in resultados if r['Ticker'] == ticker_seleccionado), None)
            if res_sel:
                col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                col_i1.markdown(f"**Estado:** {res_sel['Estado']}")
                col_i2.markdown(f"**Confianza ML:** {res_sel['Confianza ML']}")
                col_i3.markdown(f"**Riesgo Asignado:** {res_sel['Riesgo Usado']}")
                col_i4.markdown(f"**Precio Cierre:** ${res_sel['Precio']:,.2f}")

                fig_plotly = crear_grafico_donchian(
                    res_sel['Ticker'], res_sel['_df'],
                    res_sel['Stop Loss ($)'], res_sel['Take Profit ($)'],
                    res_sel['Estado']
                )
                st.plotly_chart(fig_plotly, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📋 Resultados Completos del Mercado")
            st.dataframe(
                df_res[['Ticker', 'Precio', 'Estado', 'Confianza ML', 'Half-Kelly %', 'Riesgo Usado', 'Stop Loss ($)', 'Take Profit ($)']],
                use_container_width=True
            )

            # Envío de Telegram si hay credenciales
            if telegram_token and telegram_chat_id:
                fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                msg = f"⚡ *INFORME QUANT SCANNER PRO*\n📅 `{fecha_str}`\n"
                msg += f"🌐 Macro Régimen: `{'BULLISH 🟢' if regime_ok else 'BEARISH 🔴'}`\n\n"
                
                if not df_activas.empty:
                    for _, r in df_activas.iterrows():
                        msg += f"🟢 *{r['Ticker']}* — {r['Estado']}\n• Precio: `${r['Precio']:,.2f}` | ML: `{r['Confianza ML']}` | Risk: `{r['Riesgo Usado']}`\n• SL: `${r['Stop Loss ($)']:,.2f}` | TP: `${r['Take Profit ($)']:,.2f}`\n• Operar: `{r['Unidades']} nominales` (~`${r['Capital ($)']:,.2f} USD`)\n\n"
                else:
                    msg += "⚪ Sin oportunidades activas bajo los filtros actuales."

                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                try:
                    requests.post(url, json={"chat_id": telegram_chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=5)
                    st.success("📱 Alerta enviada correctamente a Telegram.")
                except Exception as e:
                    st.warning(f"No se pudo enviar a Telegram: {e}")
