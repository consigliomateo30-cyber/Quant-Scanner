# ==============================================================================
# BLOOMBERG STYLE QUANT TERMINAL — BLACK & GOLD MINIMALIST v3.5
# Dark Gold UI | Macro Energy | Fixed Income & Real Inflation | Quant ML Scanner
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

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS CUSTOM (BLACK & GOLD MINIMALIST)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Bloomberg Quant Terminal | Black & Gold",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS Minimalista Negro & Dorado Lujo
st.markdown("""
<style>
    /* Fondo principal obsidiana */
    .stApp {
        background-color: #08080a !important;
        color: #e4e4e7 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers en blanco metálico y dorado */
    h1, h2, h3, h4 {
        color: #f4f4f5 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Subtítulos en acento dorado tenue */
    .gold-accent {
        color: #d4af37 !important;
        font-weight: 600;
    }
    
    /* Tarjetas y Contenedores */
    div[data-testid="stMetric"], .stMetric {
        background: #111115 !important;
        border: 1px solid rgba(212, 175, 55, 0.22) !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #d4af37 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        font-family: 'SF Mono', 'Roboto Mono', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a1a1aa !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    
    /* Pestañas (Tabs) estilo Bloomberg Terminal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0c0c0f;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1f1f24;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #8e8e93;
        padding: 8px 18px;
        font-weight: 500;
        font-size: 0.88rem;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a1813 !important;
        color: #d4af37 !important;
        border: 1px solid #d4af37 !important;
        font-weight: 600 !important;
    }
    
    /* Botones Dorados */
    .stButton > button {
        background: linear-gradient(180deg, #1f1a0e 0%, #12100a 100%) !important;
        color: #d4af37 !important;
        border: 1px solid #d4af37 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stButton > button:hover {
        background: #d4af37 !important;
        color: #000000 !important;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.4) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0b0e !important;
        border-right: 1px solid #1a1a20 !important;
    }
    
    /* Inputs y Textareas */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #111116 !important;
        color: #f4f4f5 !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #d4af37 !important;
        box-shadow: 0 0 6px rgba(212, 175, 55, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PANEL LATERAL CONFIGURACIÓN (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.markdown("### ⚜️ TERMINAL CONTROL")
st.sidebar.caption("Parámetros Cuantitativos de Gestión")

capital = st.sidebar.number_input("Capital Total ($ USD)", value=100000.0, step=5000.0)
riesgo_base_pct = st.sidebar.slider("Riesgo Base por Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
prob_threshold = st.sidebar.slider("Umbral Confianza ML", min_value=0.40, max_value=0.70, value=0.50, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🛡️ Filtros de Cobertura")
activar_filtro_regimen = st.sidebar.checkbox("Filtro Macro (SPY > SMA200)", value=True)
usar_kelly_dinamico = st.sidebar.checkbox("Kelly Fraccionado Dinámico", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📱 Telegram Alerts")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")

# ------------------------------------------------------------------------------
# CABECERA TERMINAL
# ------------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin-bottom:0px;'>BLOOMBERG QUANT TERMINAL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa; font-size:0.9rem;'>SISTEMA DE ANÁLISIS MACRO, RENTA FIJA Y ESCÁNER CUANTITATIVO MULTIACTIVO</p>", unsafe_allow_html=True)
with col_h2:
    st.markdown(f"<div style='text-align:right; color:#d4af37; font-family:monospace; margin-top:15px;'>{datetime.now().strftime('%Y-%m-%d | %H:%M UTC')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# DATOS MACRO Y RÉGIMEN CACHEADOS
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def obtener_datos_macro():
    tickers = {
        "WTI Oil": "CL=F",
        "Brent Oil": "BZ=F",
        "Nat Gas": "NG=F",
        "Sector Energía": "XLE",
        "Servicios Petroleros": "OIH",
        "Bono 10Y EE.UU.": "^TNX",
        "VIX Volatilidad": "^VIX",
        "Dólar DXY": "DX-Y.NYB",
        "Oro": "GC=F"
    }
    datos = {}
    for nombre, symbol in tickers.items():
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(symbol, axis=1, level=1) if symbol in df.columns.levels[1] else df.droplevel(1, axis=1)
                
                last_price = float(df['Close'].iloc[-1])
                prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else last_price
                var_pct = ((last_price - prev_price) / prev_price) * 100
                
                datos[nombre] = {
                    "symbol": symbol,
                    "precio": round(last_price, 2),
                    "var_pct": round(var_pct, 2),
                    "df": df[['Close']].tail(252)
                }
        except Exception:
            continue
    return datos

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
        
        return {
            "alcista": precio_spy >= sma200_spy,
            "precio": round(precio_spy, 2),
            "sma200": round(sma200_spy, 2),
            "msg": f"SPY ${precio_spy:.2f} vs SMA200${sma200_spy:.2f}"
        }
    except Exception as e:
        return {"alcista": True, "precio": 0, "sma200": 0, "msg": f"Error: {e}"}

# ------------------------------------------------------------------------------
# NAVEGACIÓN EN PESTAÑAS (TABS)
# ------------------------------------------------------------------------------
tab_macro, tab_rentafija, tab_scanner = st.tabs([
    "🛢️ TERMINAL MACRO & ENERGÍA",
    "🏦 RENTA FIJA & INFLACIÓN EE.UU.",
    "⚡ QUANT SCANNER PRO"
])

# ==============================================================================
# TAB 1: MACRO TERMINAL & COMMODITIES (BLACK & GOLD)
# ==============================================================================
with tab_macro:
    st.markdown("<h3 class='gold-accent'>MONITOREO MACROECONÓMICO & ENERGÍA EN TIEMPO REAL</h3>", unsafe_allow_html=True)
    
    datos_macro = obtener_datos_macro()
    
    if datos_macro:
        st.markdown("##### 🛢️ Commodities de Energía & Petróleo")
        c1, c2, c3, c4 = st.columns(4)
        
        if "WTI Oil" in datos_macro:
            d = datos_macro["WTI Oil"]
            c1.metric("WTI Crude", f"${d['precio']} USD", f"{d['var_pct']}%")
        if "Brent Oil" in datos_macro:
            d = datos_macro["Brent Oil"]
            c2.metric("Brent Crude", f"${d['precio']} USD", f"{d['var_pct']}%")
        if "Nat Gas" in datos_macro:
            d = datos_macro["Nat Gas"]
            c3.metric("Natural Gas", f"${d['precio']} USD", f"{d['var_pct']}%")
        if "Sector Energía" in datos_macro:
            d = datos_macro["Sector Energía"]
            c4.metric("Sector XLE", f"${d['precio']} USD", f"{d['var_pct']}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📊 Indicadores Globales de Riesgo y Liquidez")
        m1, m2, m3, m4 = st.columns(4)
        
        if "Bono 10Y EE.UU." in datos_macro:
            d = datos_macro["Bono 10Y EE.UU."]
            m1.metric("US 10Y Yield", f"{d['precio']}%", f"{d['var_pct']}%")
        if "VIX Volatilidad" in datos_macro:
            d = datos_macro["VIX Volatilidad"]
            m2.metric("VIX Index", f"{d['precio']}", f"{d['var_pct']}%")
        if "Dólar DXY" in datos_macro:
            d = datos_macro["Dólar DXY"]
            m3.metric("DXY Index", f"{d['precio']}", f"{d['var_pct']}%")
        if "Oro" in datos_macro:
            d = datos_macro["Oro"]
            m4.metric("Gold Futures", f"${d['precio']} USD", f"{d['var_pct']}%")

        st.markdown("---")
        st.markdown("##### 📈 Curva Comparativa Acumulada (1 Año - Normalizada %)")
        
        fig_macro = go.Figure()
        colores_gold = {"WTI Oil": "#d4af37", "Nat Gas": "#e2e8f0", "Sector Energía": "#a38340"}
        
        for k in ["WTI Oil", "Nat Gas", "Sector Energía"]:
            if k in datos_macro:
                df_c = datos_macro[k]["df"]
                norm_series = ((df_c['Close'] / df_c['Close'].iloc[0]) - 1) * 100
                fig_macro.add_trace(go.Scatter(
                    x=df_c.index, y=norm_series, mode='lines', name=k,
                    line=dict(color=colores_gold.get(k, '#d4af37'), width=2)
                ))

        fig_macro.update_layout(
            template="plotly_dark",
            paper_bgcolor="#08080a",
            plot_bgcolor="#0d0d11",
            height=400,
            xaxis=dict(showgrid=True, gridcolor="#1a1a22"),
            yaxis=dict(showgrid=True, gridcolor="#1a1a22", title="Variación %"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.08, xanchor="right", x=1)
        )
        st.plotly_chart(fig_macro, use_container_width=True)

# ==============================================================================
# TAB 2: RENTA FIJA & INFLACIÓN EE.UU.
# ==============================================================================
with tab_rentafija:
    st.markdown("<h3 class='gold-accent'>CALCULADORA DE RENTA FIJA & RASA REAL DE INFLACIÓN</h3>", unsafe_allow_html=True)
    st.caption("Ajuste por Inflación de EE.UU. con Fórmula de Fisher: Tasa Real = (1 + TNA) / (1 + Inflación) - 1")

    tnx_val = datos_macro.get("Bono 10Y EE.UU.", {}).get("precio", 4.10) if datos_macro else 4.10
    inflacion_us_est = 2.80  # Referencia Inflación Anual EE.UU. (CPI YoY)
    tasa_real_benchmark = ((1 + (tnx_val / 100)) / (1 + (inflacion_us_est / 100)) - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Bono 10Y EE.UU. (Nominal)", f"{tnx_val:.2f}% TNA")
    c2.metric("Inflación EE.UU. (CPI YoY Est.)", f"{inflacion_us_est:.2f}% Anual")
    c3.metric("Tasa Real Neta Benchmark", f"{tasa_real_benchmark:.2f}% Real")

    st.markdown("---")
    st.markdown("##### 🧮 Simulación de Bonos y T-Bills")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        monto_inv = st.number_input("Capital Invertido ($ USD):", value=10000.0, step=1000.0)
        tna_propia = st.number_input("Tasa Nominal Anual (TNA %):", value=5.25, step=0.25)
        plazo_anos = st.slider("Plazo de Madurez (Años):", min_value=0.5, max_value=30.0, value=3.0, step=0.5)

    with col_i2:
        frecuencia_cupon = st.selectbox(
            "Frecuencia de Cupones:",
            ["Al Vencimiento (Cero Cupón / T-Bill)", "Semestral (2 pagos/año)", "Anual (1 pago/año)", "Trimestral (4 pagos/año)"]
        )
        inflacion_user = st.number_input("Inflación Proyectada EE.UU. (%):", value=inflacion_us_est, step=0.1)

    pagos_por_ano = 1 if "Anual" in frecuencia_cupon else (2 if "Semestral" in frecuencia_cupon else (4 if "Trimestral" in frecuencia_cupon else 0))
    if pagos_por_ano > 0:
        n_periodos = int(plazo_anos * pagos_por_ano)
        tasa_periodo = (tna_propia / 100) / pagos_por_ano
        monto_final_nominal = monto_inv * ((1 + tasa_periodo) ** n_periodos)
    else:
        monto_final_nominal = monto_inv * (1 + (tna_propia / 100) * plazo_anos)

    ganancia_nominal_usd = monto_final_nominal - monto_inv
    tasa_real_anual = ((1 + (tna_propia / 100)) / (1 + (inflacion_user / 100)) - 1) * 100
    monto_final_real = monto_inv * ((1 + (tasa_real_anual / 100)) ** plazo_anos)
    ganancia_real_usd = monto_final_real - monto_inv

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Capital Nominal Final", f"${monto_final_nominal:,.2f} USD", f"+${ganancia_nominal_usd:,.2f}")
    r2.metric("Capital Real (Poder Compra)", f"${monto_final_real:,.2f} USD", f"+${ganancia_real_usd:,.2f}")
    r3.metric("Tasa Nominal Anual", f"{tna_propia:.2f}%")
    r4.metric("Tasa Real Neta Fisher", f"{tasa_real_anual:.2f}%")

    # Gráfico de Proyección
    anos_eje = np.linspace(0, plazo_anos, 50)
    curva_nominal = monto_inv * ((1 + (tna_propia / 100)) ** anos_eje)
    curva_real = monto_inv * ((1 + (tasa_real_anual / 100)) ** anos_eje)

    fig_rf = go.Figure()
    fig_rf.add_trace(go.Scatter(x=anos_eje, y=curva_nominal, mode='lines', name='Curva Nominal ($ USD)', line=dict(color='#d4af37', width=2.5)))
    fig_rf.add_trace(go.Scatter(x=anos_eje, y=curva_real, mode='lines', name='Curva Real (Ajustada Inflación)', line=dict(color='#e2e8f0', width=2, dash='dash')))

    fig_rf.update_layout(
        title="Proyección Temporal de Retorno Nominal vs Real",
        xaxis_title="Tiempo (Años)",
        yaxis_title="Monto ($ USD)",
        template="plotly_dark",
        paper_bgcolor="#08080a",
        plot_bgcolor="#0d0d11",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_rf, use_container_width=True)

# ==============================================================================
# TAB 3: QUANT SCANNER PRO (26 ACTIVOS + ML + HALF-KELLY)
# ==============================================================================
with tab_scanner:
    st.markdown("<h3 class='gold-accent'>ESCÁNER CUANTITATIVO Y SEÑALES DE MACHINE LEARNING</h3>", unsafe_allow_html=True)
    
    tickers_input = st.text_area(
        "Watchlist de Activos Monitoreados (26 activos predeterminados):",
        value="SPY, QQQ, NVDA, AAPL, MSFT, AMZN, TSLA, META, BTC-USD, MELI, OKLO, QCOM, AVGO, YPF, TGS, MU, NU, ASML, IREN, VIST, VT, PAMP, IBIT, TSM, SPCX, BRK-B",
        height=85
    )

    regimen_info = obtener_regimen_mercado()
    if activar_filtro_regimen:
        if regimen_info['alcista']:
            st.success(f"🌐 **Régimen Macro Favorables (BULLISH):** {regimen_info['msg']}. Mercado apto para posiciones de Breakout.")
        else:
            st.warning(f"⚠️ **Régimen Macro Cauteloso (BEARISH):** {regimen_info['msg']}. Compras bloqueadas para proteger capital.")

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

    def crear_grafico_donchian(ticker, df_hist, precio_sl, precio_tp, estado):
        fig = go.Figure()

        # Precio
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], mode='lines', name='Precio Cierre', line=dict(color='#e4e4e7', width=2)))
        # Donchian High (Dorado)
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Donchian_High'], mode='lines', name='Donchian High (20d)', line=dict(color='#d4af37', width=1.5, dash='dash')))
        # Donchian Low (Gris)
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Donchian_Low'], mode='lines', name='Donchian Low (10d)', line=dict(color='#ef4444', width=1.5, dash='dash')))

        if "COMPRA" in estado or "MANTENER" in estado:
            last_date = df_hist.index[-1]
            fig.add_trace(go.Scatter(x=[last_date], y=[precio_tp], mode='markers+text', name='Take Profit', marker=dict(color='#d4af37', size=10, symbol='triangle-up'), text=[f"TP: ${precio_tp:,.2f}"], textposition="top center"))
            fig.add_trace(go.Scatter(x=[last_date], y=[precio_sl], mode='markers+text', name='Stop Loss', marker=dict(color='#ef4444', size=10, symbol='triangle-down'), text=[f"SL: ${precio_sl:,.2f}"], textposition="bottom center"))

        fig.update_layout(
            title=f"Evolución Técnica y Canales de Donchian — {ticker}",
            xaxis_title="Fecha", yaxis_title="Precio ($ USD)", template="plotly_dark",
            paper_bgcolor="#08080a", plot_bgcolor="#0d0d11",
            height=400, margin=dict(l=30, r=30, t=40, b=30)
        )
        return fig

    if st.button("🚀 EJECUTAR ESCÁNER CUANTITATIVO", type="primary", use_container_width=True):
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
                col2.metric("Oportunidades ML", len(df_activas))
                col3.metric("Capital Asignado", f"${df_activas['Capital ($)'].sum():,.2f} USD")

                st.markdown("---")
                st.markdown("##### 🎯 Oportunidades Activas Aprobadas")
                if not df_activas.empty:
                    st.dataframe(
                        df_activas[['Ticker', 'Precio', 'Estado', 'Confianza ML', 'Half-Kelly %', 'Riesgo Usado', 'Stop Loss ($)', 'Take Profit ($)', 'Unidades', 'Capital ($)']],
                        use_container_width=True
                    )
                else:
                    st.info("Sin señales de compra activas que superen los filtros en este momento.")

                st.markdown("---")
                st.markdown("##### 📊 Gráficos Interactivos")
                
                opciones_tickers = [r['Ticker'] for r in resultados]
                ticker_seleccionado = st.selectbox("🔍 Seleccionar activo a analizar:", opciones_tickers)
                
                res_sel = next((r for r in resultados if r['Ticker'] == ticker_seleccionado), None)
                if res_sel:
                    c_i1, c_i2, c_i3, c_i4 = st.columns(4)
                    c_i1.markdown(f"**Estado:** {res_sel['Estado']}")
                    c_i2.markdown(f"**Confianza ML:** {res_sel['Confianza ML']}")
                    c_i3.markdown(f"**Riesgo Asignado:** {res_sel['Riesgo Usado']}")
                    c_i4.markdown(f"**Precio Cierre:** ${res_sel['Precio']:,.2f}")

                    fig_plotly = crear_grafico_donchian(res_sel['Ticker'], res_sel['_df'], res_sel['Stop Loss ($)'], res_sel['Take Profit ($)'], res_sel['Estado'])
                    st.plotly_chart(fig_plotly, use_container_width=True)

                st.markdown("---")
                st.markdown("##### 📋 Resultados Completos del Mercado")
                st.dataframe(df_res[['Ticker', 'Precio', 'Estado', 'Confianza ML', 'Half-Kelly %', 'Riesgo Usado', 'Stop Loss ($)', 'Take Profit ($)']], use_container_width=True)

                if telegram_token and telegram_chat_id:
                    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    msg = f"⚡ *INFORME QUANT TERMINAL PRO*\n📅 `{fecha_str}`\n"
                    msg += f"🌐 Macro Régimen: `{'BULLISH 🟢' if regime_ok else 'BEARISH 🔴'}`\n\n"
                    
                    if not df_activas.empty:
                        for _, r in df_activas.iterrows():
                            msg += f"🟢 *{r['Ticker']}* — {r['Estado']}\n• Precio: `${r['Precio']:,.2f}` | ML: `{r['Confianza ML']}` | Risk: `{r['Riesgo Usado']}`\n• SL: `${r['Stop Loss ($)']:,.2f}` | TP: `${r['Take Profit ($)']:,.2f}`\n• Operar: `{r['Unidades']} nominales` (~`${r['Capital ($)']:,.2f} USD`)\n\n"
                    else:
                        msg += "⚪ Sin oportunidades activas bajo los filtros actuales."

                    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                    try:
                        requests.post(url, json={"chat_id": telegram_chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=5)
                        st.success("📱 Alerta enviada a Telegram.")
                    except Exception as e:
                        st.warning(f"Error Telegram: {e}")
