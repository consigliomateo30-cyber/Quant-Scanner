# ==============================================================================
# QUANT SCANNER PRO — TERMINAL INSTITUCIONAL & GENERADOR DE INFORMES v6.0
# Sistema de Escaneo Cuantitativo, Riesgo Monte Carlo y Generación de Factsheet PDF/HTML
# ==============================================================================

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import math
import requests
import base64
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS CUSTOM
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Scanner Pro | Terminal de Gestión",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #08080a !important;
        color: #e4e4e7 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    h1, h2, h3, h4 {
        color: #f4f4f5 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    .gold-accent {
        color: #d4af37 !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
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
        font-size: 1.4rem !important;
        font-family: 'SF Mono', 'Roboto Mono', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a1a1aa !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    
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
        font-size: 0.85rem;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a1813 !important;
        color: #d4af37 !important;
        border: 1px solid #d4af37 !important;
        font-weight: 600 !important;
    }
    
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
    
    .news-card {
        background: #111116;
        border: 1px solid #27272a;
        border-left: 3px solid #d4af37;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0b0b0e !important;
        border-right: 1px solid #1a1a20 !important;
    }
    
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #111116 !important;
        color: #f4f4f5 !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# INICIALIZACIÓN DE MEMORIA DE SESIÓN (SESSION STATE)
# ------------------------------------------------------------------------------
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None

ALL_TICKERS_DEFAULT = [
    "SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", 
    "BTC-USD", "MELI", "OKLO", "QCOM", "AVGO", "YPF", "TGS", "MU", 
    "NU", "ASML", "IREN", "VIST", "VT", "PAMP", "IBIT", "TSM", "SPCX", "BRK-B"
]

# ------------------------------------------------------------------------------
# PANEL LATERAL (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ QUANT SCANNER CONTROL")
st.sidebar.caption("Parámetros de Portafolio y Riesgo")

capital = st.sidebar.number_input("Capital Portfolio ($ USD)", value=100000.0, step=5000.0)
riesgo_base_pct = st.sidebar.slider("Riesgo Base por Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
prob_threshold = st.sidebar.slider("Umbral Confianza ML (p-hat)", min_value=0.40, max_value=0.70, value=0.50, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🛡️ Cobertura & Régimen Macro")
activar_filtro_regimen = st.sidebar.checkbox("Filtro Régimen Macro (SPY > SMA200)", value=True)
usar_kelly_dinamico = st.sidebar.checkbox("Kelly Fraccionado Dinámico", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📱 Telegram Gateway")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")

# ------------------------------------------------------------------------------
# CABECERA PRINCIPAL
# ------------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin-bottom:0px;'>QUANT SCANNER PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa; font-size:0.88rem;'>TERMINAL INSTITUCIONAL DE GESTIÓN CUANTITATIVA Y RIESGO DE CARTERA</p>", unsafe_allow_html=True)
with col_h2:
    st.markdown(f"<div style='text-align:right; color:#d4af37; font-family:monospace; margin-top:15px;'>{datetime.now().strftime('%Y-%m-%d | %H:%M UTC')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES Y DE DATOS CACHEADOS
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def obtener_datos_macro():
    tickers = {
        "WTI Oil": "CL=F", "Brent Oil": "BZ=F", "Nat Gas": "NG=F",
        "Sector Energía": "XLE", "Servicios Petroleros": "OIH",
        "Bono 10Y EE.UU.": "^TNX", "VIX Volatilidad": "^VIX",
        "Dólar DXY": "DX-Y.NYB", "Oro": "GC=F"
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
                    "symbol": symbol, "precio": round(last_price, 2),
                    "var_pct": round(var_pct, 2), "df": df[['Close']].tail(252)
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
            "msg": f"SPY ${precio_spy:.2f} vs SMA200 ${sma200_spy:.2f}"
        }
    except Exception as e:
        return {"alcista": True, "precio": 0, "sma200": 0, "msg": f"Error: {e}"}

@st.cache_data(ttl=1800)
def obtener_noticias_ticker(ticker):
    try:
        tk = yf.Ticker(ticker)
        news = tk.news
        if news and len(news) > 0:
            return news[:3]
    except Exception:
        pass
    return []

@st.cache_data(ttl=3600)
def obtener_fundamentales_ticker(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        gross_m = info.get("grossMargins", None)
        pricing_power = "ALTO 🟢" if (gross_m or 0) >= 0.40 else ("MEDIO 🟡" if (gross_m or 0) >= 0.20 else "N/A ⚪")
        
        mcap = info.get("marketCap", 0)
        fcf = info.get("freeCashflow", 0)
        
        return {
            "Ticker": ticker,
            "Sector": info.get("sector", "N/A"),
            "Market Cap ($B)": round(mcap / 1e9, 2) if mcap else "N/A",
            "Forward P/E": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A",
            "Gross Margin (%)": f"{round(gross_m * 100, 1)}%" if gross_m is not None else "N/A",
            "Profit Margin (%)": f"{round(info.get('profitMargins', 0) * 100, 1)}%" if info.get("profitMargins") is not None else "N/A",
            "Debt / Equity": round(info.get("debtToEquity", 0), 2) if info.get("debtToEquity") else "N/A",
            "Free Cash Flow ($B)": round(fcf / 1e9, 2) if fcf else "N/A",
            "Pricing Power Index": pricing_power
        }
    except Exception:
        return {
            "Ticker": ticker, "Sector": "N/A", "Market Cap ($B)": "N/A",
            "Forward P/E": "N/A", "Gross Margin (%)": "N/A", "Profit Margin (%)": "N/A",
            "Debt / Equity": "N/A", "Free Cash Flow ($B)": "N/A", "Pricing Power Index": "N/A ⚪"
        }

# ------------------------------------------------------------------------------
# GENERADOR DE INFORME HTML EJECUTIVO (ESTILO DOCUMENTO DE INVERSIÓN)
# ------------------------------------------------------------------------------
def generar_informe_html_ejecutivo(resultados_scan, capital_total, regimen_data):
    fecha_actual = datetime.now().strftime("%d de %B de %Y — %H:%M UTC")
    df_res = pd.DataFrame(resultados_scan).sort_values(by=["_prioridad", "Var 1D %"], ascending=[True, False])
    df_aprobados = df_res[df_res['_prioridad'] == 1]
    df_rechazados = df_res[df_res['_prioridad'] > 1]
    
    capital_asignado = df_aprobados['Allocation ($)'].sum() if not df_aprobados.empty else 0.0
    liquidez_restante = capital_total - capital_asignado

    # Construcción de filas de tabla aprobados
    filas_aprobados_html = ""
    if not df_aprobados.empty:
        for _, r in df_aprobados.iterrows():
            filas_aprobados_html += f"""
            <tr>
                <td><strong>{r['Ticker']}</strong></td>
                <td>${r['Precio']:,.2f} ({r['Var 1D %']}%)</td>
                <td><span style="color: #22c55e; font-weight: bold;">{r['Estado']}</span></td>
                <td>{r['p-hat (ML)']}</td>
                <td>${r['Stop Loss ($)']:,.2f}</td>
                <td>${r['Take Profit ($)']:,.2f}</td>
                <td><strong>{r['Nominales']}</strong> u.</td>
                <td><strong>${r['Allocation ($)']:,.2f}</strong></td>
            </tr>
            """
    else:
        filas_aprobados_html = "<tr><td colspan='8' style='text-align:center;'>No se registran posiciones de compra activas en la fecha.</td></tr>"

    # Construcción de filas de tabla rechazados
    filas_rechazados_html = ""
    for _, r in df_rechazados.iterrows():
        razon = "Filtro ML p-hat bajo umbral" if "META-FILTER" in r['Estado'] else ("Mercado en régimen bajista" if "MACRO" in r['Estado'] else "Sin quiebre técnico activo")
        filas_rechazados_html += f"""
        <tr>
            <td><strong>{r['Ticker']}</strong></td>
            <td>${r['Precio']:,.2f}</td>
            <td><span style="color: #eab308;">{r['Estado']}</span></td>
            <td>{r['p-hat (ML)']}</td>
            <td>{razon}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Quant Scanner Pro — Executive Factsheet</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                background-color: #ffffff;
                color: #18181b;
                margin: 0;
                padding: 40px;
                line-height: 1.6;
            }}
            .header {{
                border-bottom: 3px solid #d4af37;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                color: #09090b;
                font-size: 26px;
                letter-spacing: -0.5px;
            }}
            .header p {{
                margin: 5px 0 0 0;
                color: #71717a;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .meta-bar {{
                background-color: #f4f4f5;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 30px;
                display: flex;
                justify-content: space-between;
                font-size: 13px;
            }}
            .index-box {{
                background-color: #fafafa;
                border: 1px solid #e4e4e7;
                border-left: 4px solid #d4af37;
                padding: 20px;
                margin-bottom: 35px;
                border-radius: 4px;
            }}
            .index-box h3 {{
                margin-top: 0;
                color: #09090b;
                font-size: 16px;
            }}
            .section-title {{
                color: #09090b;
                border-bottom: 1px solid #e4e4e7;
                padding-bottom: 8px;
                margin-top: 35px;
                margin-bottom: 15px;
                font-size: 18px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 25px;
                font-size: 12px;
            }}
            th, td {{
                border: 1px solid #e4e4e7;
                padding: 10px 12px;
                text-align: left;
            }}
            th {{
                background-color: #18181b;
                color: #f4f4f5;
                font-weight: 600;
            }}
            tr:nth-child(even) {{
                background-color: #fcfcfc;
            }}
            .footer {{
                margin-top: 50px;
                border-top: 1px solid #e4e4e7;
                padding-top: 20px;
                font-size: 11px;
                color: #a1a1aa;
                text-align: center;
            }}
            .badge-green {{
                background-color: #dcfce7;
                color: #166534;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            @media print {{
                body {{ padding: 0; }}
                .no-print {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>QUANT SCANNER PRO — FACTSHEET EJECUTIVO</h1>
            <p>Informe Estratégico de Asignación Cuantitativa & Auditoría de Riesgo de Cartera</p>
        </div>

        <div class="meta-bar">
            <div><strong>Fecha de Emisión:</strong> {fecha_actual}</div>
            <div><strong>Capital Total AUM:</strong> ${capital_total:,.2f} USD</div>
            <div><strong>Asignado en Largo:</strong> ${capital_asignado:,.2f} USD</div>
            <div><strong>Régimen Macro (SPY):</strong> {'BULLISH 🟢' if regimen_data['alcista'] else 'BEARISH 🔴'}</div>
        </div>

        <div class="index-box">
            <h3>📑 ÍNDICE Y FUNDAMENTOS DEL MODELO CUANTITATIVO</h3>
            <ol>
                <li><strong>Fundamento Algorítmico (Donchian Volatility Breakout):</strong> Se utilizan canales de ruptura de 20 días de máximo y 10 días de mínimo para capturar tendencias de alta inercia institucional.</li>
                <li><strong>Triple Barrier Method (Marcos López de Prado):** En lugar de medir retornos a plazos fijos, cada operación se audita mediante barreras horizontales de Stop Loss (-1.2 ATR) y Take Profit (+1.8 ATR) a 5 días de horizonte.</li>
                <li><strong>Filtro Secundario con Machine Learning (Random Forest Meta-Labeling):</strong> Un modelo de Inteligencia Artificial evalúa variables de microestructura (RSI, Volatilidad 20d, Ratio de Volumen) para estimar la probabilidad real <code>p-hat</code> de éxito y desestimar falsos quiebres (whipsaws).</li>
                <li><strong>Gestión Monetaria (Half-Kelly Dinámico):</strong> El tamaño de la posición se modula dinámicamente según el grado de certidumbre del modelo para prevenir la ruina del capital y maximizar la tasa de crecimiento logarítmico.</li>
            </ol>
        </div>

        <div class="section-title">1. Resumen Ejecutivo de Oportunidades Aprobadas</div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Precio (Var 1D)</th>
                    <th>Estado</th>
                    <th>Confianza ML (p-hat)</th>
                    <th>Stop Loss</th>
                    <th>Take Profit</th>
                    <th>Nominales</th>
                    <th>Capital Asignado</th>
                </tr>
            </thead>
            <tbody>
                {filas_aprobados_html}
            </tbody>
        </table>

        <div class="section-title">2. Activos Evaluados y Razón de Descarte Táctico</div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Precio Cierre</th>
                    <th>Estado Algorítmico</th>
                    <th>p-hat (ML)</th>
                    <th>Justificación del Descarte / Neutralidad</th>
                </tr>
            </thead>
            <tbody>
                {filas_rechazados_html}
            </tbody>
        </table>

        <div class="section-title">3. Tesis Táctica y Recomendación Institucional</div>
        <p>
            Bajo las condiciones actuales de liquidez y volatilidad, el portafolio mantiene una exposición activa del <strong>{(capital_asignado/capital_total)*100:.1f}%</strong> del AUM total, conservando un <strong>{(liquidez_restante/capital_total)*100:.1f}%</strong> en liquidez/Treasuries libres de riesgo. Se recomienda la ejecución estricta de las órdenes con los niveles de Stop Loss indicados para preservar la asimetría positiva del perfil de riesgo.
        </p>

        <div class="footer">
            <p>QUANT SCANNER PRO — DOCUMENTO CONFIDENCIAL GENERADO PARA COMITÉ DE INVERSIÓN Y GESTIÓN PRIVADA</p>
            <p>Para guardar este informe en PDF, presione Ctrl + P (o Cmd + P en Mac) y seleccione 'Guardar como PDF'.</p>
        </div>
    </body>
    </html>
    """
    return html_content

# ------------------------------------------------------------------------------
# NAVEGACIÓN EN PESTAÑAS (TABS)
# ------------------------------------------------------------------------------
tab_scanner, tab_reporte, tab_fundamentales, tab_macro, tab_rentafija, tab_risk = st.tabs([
    "⚡ ESCÁNER CUANTITATIVO PRO",
    "📄 INFORME EJECUTIVO (FACTSHEET)",
    "🏛️ ANÁLISIS FUNDAMENTAL & RIESGO",
    "🛢️ MACRO & COMMODITIES",
    "🏦 RENTA FIJA & INFLACIÓN EE.UU.",
    "🛡️ RIESGO DE CARTERA & MONTE CARLO"
])

# ==============================================================================
# TAB 1: ESCÁNER CUANTITATIVO PRO
# ==============================================================================
with tab_scanner:
    st.markdown("<h3 class='gold-accent'>MOTOR DE GENERACIÓN DE ALFA & META-LABELING ML</h3>", unsafe_allow_html=True)
    
    st.markdown("##### 🎯 Selección de Activos a Escanear")
    modo_escaneo = st.radio(
        "Modo de Análisis:",
        ["Watchlist completa (26 activos predeterminados)", "Selección personalizada (Elegir activos)"],
        horizontal=True
    )

    if modo_escaneo == "Watchlist completa (26 activos predeterminados)":
        tickers_a_procesar = ALL_TICKERS_DEFAULT
    else:
        tickers_a_procesar = st.multiselect(
            "Seleccioná los activos a evaluar:",
            ALL_TICKERS_DEFAULT,
            default=["SPY", "NVDA", "YPF", "BTC-USD", "AAPL", "MSFT"]
        )

    regimen_info = obtener_regimen_mercado()
    if activar_filtro_regimen:
        if regimen_info['alcista']:
            st.success(f"🌐 **Régimen Macro (BULLISH):** {regimen_info['msg']}. Mercado apto para posiciones de compra.")
        else:
            st.warning(f"⚠️ **Régimen Macro (BEARISH):** {regimen_info['msg']}. Compras bloqueadas preventivamente.")

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

            retorno_diario = float((precio_cierre - float(df_clean['Close'].iloc[-2])) / float(df_clean['Close'].iloc[-2])) * 100 if len(df_clean) > 1 else 0.0

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
                estado_str = "⛔ MACRO REGIME REJECT"
                prioridad = 4
            elif primary_signal == 1 and ml_prob >= threshold_val:
                estado_str = "🟢 LONG ALPHA ENTRY" if primary_entry == 1 else "🔵 HOLD POSITION"
                prioridad = 1
            elif primary_signal == 1 and ml_prob < threshold_val:
                estado_str = "🟡 META-FILTER REJECT"
                prioridad = 2
            else:
                estado_str = "⚪ LIQUID / NO SIGNAL"
                prioridad = 3

            return {
                "Ticker": ticker.upper(),
                "Precio": round(precio_cierre, 2),
                "Var 1D %": round(retorno_diario, 2),
                "Estado": estado_str,
                "p-hat (ML)": f"{ml_prob*100:.1f}%",
                "Half-Kelly": f"{half_kelly_frac*100:.1f}%",
                "Risk Budget": f"{riesgo_efectivo_pct:.2f}%",
                "Stop Loss ($)": round(precio_sl, 2) if primary_signal == 1 else 0.0,
                "Take Profit ($)": round(precio_tp, 2) if primary_signal == 1 else 0.0,
                "Nominales": unidades if prioridad == 1 else 0,
                "Allocation ($)": round(monto_total_usd, 2) if prioridad == 1 else 0.0,
                "_df": df_clean.tail(120),
                "_returns": df_clean['Close'].pct_change().dropna(),
                "_prioridad": prioridad,
                "_prob_raw": ml_prob
            }
        except Exception:
            return None

    def crear_grafico_donchian(ticker, df_hist, precio_sl, precio_tp, estado):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], mode='lines', name='Precio Cierre', line=dict(color='#e4e4e7', width=2)))
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Donchian_High'], mode='lines', name='Donchian High (20d)', line=dict(color='#d4af37', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Donchian_Low'], mode='lines', name='Donchian Low (10d)', line=dict(color='#ef4444', width=1.5, dash='dash')))

        if "LONG" in estado or "HOLD" in estado:
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

    if st.button("🚀 EJECUTAR ANÁLISIS DE MERCADO", type="primary", use_container_width=True):
        if not tickers_a_procesar:
            st.error("Seleccioná al menos un activo válido.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            resultados = []
            regime_ok = regimen_info['alcista']

            for i, tk in enumerate(tickers_a_procesar):
                status_text.text(f"Analizando {tk} ({i+1}/{len(tickers_a_procesar)})...")
                res = procesar_activo(tk, capital, riesgo_base_pct, prob_threshold, regime_ok, usar_kelly_dinamico)
                if res:
                    resultados.append(res)
                progress_bar.progress((i + 1) / len(tickers_a_procesar))

            status_text.empty()
            progress_bar.empty()

            if resultados:
                st.session_state['scan_results'] = resultados

    if st.session_state['scan_results'] is not None:
        resultados = st.session_state['scan_results']
        df_res = pd.DataFrame(resultados).sort_values(by=["_prioridad", "Var 1D %"], ascending=[True, False])
        df_activas = df_res[df_res['_prioridad'] == 1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Activos Evaluados", len(resultados))
        col2.metric("Oportunidades Alpha (p-hat > TH)", len(df_activas))
        col3.metric("Capital Asignado Total", f"${df_activas['Allocation ($)'].sum():,.2f} USD")

        st.markdown("---")
        st.markdown("##### 🎯 Señales Activas de Compra / Mantener")
        if not df_activas.empty:
            st.dataframe(
                df_activas[['Ticker', 'Precio', 'Var 1D %', 'Estado', 'p-hat (ML)', 'Half-Kelly', 'Risk Budget', 'Stop Loss ($)', 'Take Profit ($)', 'Nominales', 'Allocation ($)']],
                use_container_width=True
            )
        else:
            st.info("Sin oportunidades activas de compra bajo los filtros actuales.")

        st.markdown("---")
        st.markdown("##### 📊 Inspección Técnica Individual (Seleccionar Activo)")
        
        opciones_tickers = [r['Ticker'] for r in resultados]
        ticker_seleccionado = st.selectbox("🔍 Seleccionar activo para graficar:", opciones_tickers)
        
        res_sel = next((r for r in resultados if r['Ticker'] == ticker_seleccionado), None)
        if res_sel:
            c_i1, c_i2, c_i3, c_i4 = st.columns(4)
            c_i1.markdown(f"**Estado:** {res_sel['Estado']}")
            c_i2.markdown(f"**Probabilidad p-hat:** {res_sel['p-hat (ML)']}")
            c_i3.markdown(f"**Risk Budget:** {res_sel['Risk Budget']}")
            c_i4.markdown(f"**Precio Cierre:** ${res_sel['Precio']:,.2f} (`{res_sel['Var 1D %']}%`)")

            fig_plotly = crear_grafico_donchian(res_sel['Ticker'], res_sel['_df'], res_sel['Stop Loss ($)'], res_sel['Take Profit ($)'], res_sel['Estado'])
            st.plotly_chart(fig_plotly, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📋 Tabla General de Resultados de Mercado")
        st.dataframe(df_res[['Ticker', 'Precio', 'Var 1D %', 'Estado', 'p-hat (ML)', 'Half-Kelly', 'Risk Budget', 'Stop Loss ($)', 'Take Profit ($)']], use_container_width=True)

        # NOTICIAS EN TIEMPO REAL
        st.markdown("---")
        st.markdown("##### 📰 NOTICIAS EN TIEMPO REAL & PRICING POWER (TOP MOVERS)")

        top_movers = df_res.sort_values(by="Var 1D %", ascending=False).head(4)
        cols_noticias = st.columns(len(top_movers))
        
        for idx, (_, row) in enumerate(top_movers.iterrows()):
            tk_news = row['Ticker']
            news_list = obtener_noticias_ticker(tk_news)
            
            with cols_noticias[idx]:
                st.markdown(f"**{tk_news}** (`{row['Var 1D %']}%`)")
                if news_list:
                    for item in news_list[:2]:
                        title = item.get("title", "Noticia sin título")
                        link = item.get("link", "#")
                        publisher = item.get("publisher", "Market News")
                        st.markdown(f"""
                        <div class="news-card">
                            <a href="{link}" target="_blank" style="color:#d4af37; text-decoration:none; font-weight:600; font-size:0.82rem;">{title[:75]}...</a>
                            <p style="color:#a1a1aa; font-size:0.7rem; margin-top:4px;">Fuente: {publisher}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("Sin noticias recientes.")

# ==============================================================================
# TAB 2: INFORME EJECUTIVO (FACTSHEET PDF/HTML)
# ==============================================================================
with tab_reporte:
    st.markdown("<h3 class='gold-accent'>INFORME EJECUTIVO & FACTSHEET INSTITUCIONAL</h3>", unsafe_allow_html=True)
    st.caption("Generación de Documentos de Inversión Formateados para Comités de Riesgo y Clientes Privados")

    if st.session_state['scan_results'] is None:
        st.info("👈 Por favor ejecutá primero el escáner en la pestaña '⚡ ESCÁNER CUANTITATIVO PRO' para consolidar los datos del informe.")
    else:
        st.success("✅ Informe consolidado listo para lectura y exportación.")
        
        regime_data = obtener_regimen_mercado()
        html_doc = generar_informe_html_ejecutivo(st.session_state['scan_results'], capital, regime_data)

        st.markdown("##### 📄 Previsualización del Documento Ejecutivo")
        components.html(html_doc, height=650, scrolling=True)

        st.markdown("---")
        st.markdown("##### 📥 Opciones de Descarga e Impresión")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📄 DESCARGAR FACTSHEET (DOCUMENTO HTML)",
                data=html_doc,
                file_name=f"Quant_Executive_Report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
        with col_d2:
            st.info("💡 **Para Guardar como PDF:** Abrí el documento HTML descargado y presioná `Ctrl + P` (o `Cmd + P` en Mac) para elegir 'Guardar como PDF'.")

# ==============================================================================
# TAB 3: ANÁLISIS FUNDAMENTAL & RIESGO
# ==============================================================================
with tab_fundamentales:
    st.markdown("<h3 class='gold-accent'>ANÁLISIS FUNDAMENTAL & MARCO DE RIESGO</h3>", unsafe_allow_html=True)
    st.caption("Evaluación de Pricing Power, Cobertura de Deuda y Salud Financiera Corporativa")

    st.markdown("""
    #### 🏛️ Los 4 Pilares de Auditoría:
    1. **Pricing Power & Margen Bruto:** Capacidad de la empresa para trasladar la inflación a precios sin perder volumen (Margen Bruto > 40%).
    2. **Cobertura de Deuda & Solvencia:** Relación Deuda/Patrimonio (Debt to Equity < 1.5) para evitar riesgos en entornos de tasas altas.
    3. **Free Cash Flow Yield:** Conversión de beneficios contables en flujo de caja libre real para reinversión.
    4. **Sensibilidad Macro:** Resistencia ante cambios en el Dólar (DXY), Petróleo y Tasa 10Y.
    """)

    st.markdown("---")
    st.markdown("##### 📋 Matriz de Indicadores Fundamentales")

    if st.button("📊 Cargar/Actualizar Datos Fundamentales de la Watchlist", type="primary"):
        fund_list = []
        progress_f = st.progress(0)
        for idx, tk_f in enumerate(ALL_TICKERS_DEFAULT[:15]):
            res_f = obtener_fundamentales_ticker(tk_f)
            if res_f:
                fund_list.append(res_f)
            progress_f.progress((idx + 1) / 15)
        progress_f.empty()

        if fund_list:
            st.session_state['fund_table'] = pd.DataFrame(fund_list)

    if 'fund_table' in st.session_state:
        st.dataframe(st.session_state['fund_table'], use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔍 Profiler Fundamental e Inspección por Activo")
    
    tk_selected = st.selectbox("Elegí un activo para auditar sus métricas fundamentales:", ALL_TICKERS_DEFAULT)
    fund_single = obtener_fundamentales_ticker(tk_selected)
    
    if fund_single:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pricing Power Index", fund_single["Pricing Power Index"])
        c2.metric("Gross Margin (%)", fund_single["Gross Margin (%)"])
        c3.metric("Free Cash Flow ($B)", f"${fund_single['Free Cash Flow ($B)']} B")
        c4.metric("Forward P/E", fund_single["Forward P/E"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sector", fund_single["Sector"])
        c6.metric("Market Cap ($B)", f"${fund_single['Market Cap ($B)']} B")
        c7.metric("Profit Margin (%)", fund_single["Profit Margin (%)"])
        c8.metric("Debt / Equity", fund_single["Debt / Equity"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"##### 📰 Noticias Corporativas & Catalizadores — {tk_selected}")
    single_news = obtener_noticias_ticker(tk_selected)
    if single_news:
        for item in single_news:
            st.markdown(f"""
            <div class="news-card">
                <a href="{item.get('link', '#')}" target="_blank" style="color:#d4af37; text-decoration:none; font-weight:600; font-size:0.88rem;">{item.get('title', 'Noticia sin título')}</a>
                <p style="color:#a1a1aa; font-size:0.72rem; margin-top:4px;">Fuente: {item.get('publisher', 'Financial News')}</p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: MACRO & COMMODITIES
# ==============================================================================
with tab_macro:
    st.markdown("<h3 class='gold-accent'>MONITOREO MACROECONÓMICO & SECTOR ENERGÍA</h3>", unsafe_allow_html=True)
    
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
            height=380,
            xaxis=dict(showgrid=True, gridcolor="#1a1a22"),
            yaxis=dict(showgrid=True, gridcolor="#1a1a22", title="Variación %"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.08, xanchor="right", x=1)
        )
        st.plotly_chart(fig_macro, use_container_width=True)

# ==============================================================================
# TAB 5: RENTA FIJA & INFLACIÓN EE.UU.
# ==============================================================================
with tab_rentafija:
    st.markdown("<h3 class='gold-accent'>CALCULADORA DE RENTA FIJA & RENTABILIDAD REAL</h3>", unsafe_allow_html=True)
    st.caption("Ajuste por Inflación de EE.UU. con Fórmula de Fisher: Tasa Real = (1 + TNA) / (1 + Inflación) - 1")

    tnx_val = datos_macro.get("Bono 10Y EE.UU.", {}).get("precio", 4.10) if datos_macro else 4.10
    inflacion_us_est = 2.80
    tasa_real_benchmark = ((1 + (tnx_val / 100)) / (1 + (inflacion_us_est / 100)) - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("US 10Y Benchmark (Nominal)", f"{tnx_val:.2f}% TNA")
    c2.metric("US CPI YoY (Inflation Est.)", f"{inflacion_us_est:.2f}% Anual")
    c3.metric("Net Real Rate Benchmark", f"{tasa_real_benchmark:.2f}% Real")

    st.markdown("---")
    st.markdown("##### 🧮 Simulación de Bonos, T-Bills y Estructura Temporal")

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
# TAB 6: RIESGO DE CARTERA & MONTE CARLO
# ==============================================================================
with tab_risk:
    st.markdown("<h3 class='gold-accent'>MOTOR DE RIESGO INSTITUCIONAL & SIMULACIÓN MONTE CARLO</h3>", unsafe_allow_html=True)
    st.caption("Métricas Avanzadas de Riesgo de Cola (VaR / CVaR 99%), Deflated Sharpe Ratio (DSR) & Simulación Estocástica")

    @st.cache_data(ttl=3600)
    def obtener_retornos_base_cartera():
        tickers_base = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "YPF", "VT"]
        try:
            df = yf.download(tickers_base, period="1y", interval="1d", progress=False)['Close']
            return df.pct_change().dropna()
        except Exception:
            return pd.DataFrame()

    scan_res = st.session_state.get('scan_results', [])

    if scan_res:
        returns_dict = {r['Ticker']: r['_returns'] for r in scan_res if r['_returns'] is not None and len(r['_returns']) > 50}
        df_returns = pd.DataFrame(returns_dict).dropna()
        origen_str = "Señales del Escáner Cuantitativo"
    else:
        df_returns = obtener_retornos_base_cartera()
        origen_str = "Cartera Benchmark Base (SPY, QQQ, NVDA, AAPL, MSFT, YPF, VT)"

    if not df_returns.empty:
        st.info(f"📊 Evaluando métricas de riesgo sobre: **{origen_str}**")
        port_returns = df_returns.mean(axis=1)

        mean_ret = port_returns.mean()
        std_ret = port_returns.std()
        
        sharpe_ann = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        neg_ret = port_returns[port_returns < 0]
        downside_std = neg_ret.std()
        sortino_ann = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        var_99_pct = abs(np.percentile(port_returns, 1))
        var_99_usd = var_99_pct * capital

        tail_losses = port_returns[port_returns <= -var_99_pct]
        cvar_99_pct = abs(tail_losses.mean()) if len(tail_losses) > 0 else var_99_pct
        cvar_99_usd = cvar_99_pct * capital

        skew_val = port_returns.skew()
        kurt_val = port_returns.kurtosis()
        N_obs = len(port_returns)
        
        dsr_std = np.sqrt((1 - skew_val * sharpe_ann + ((kurt_val - 1) / 4) * (sharpe_ann ** 2)) / (N_obs - 1)) if N_obs > 1 else 1.0
        dsr_prob = max(0.01, min(0.99, 1.0 - (0.5 * (1.0 + math.erf(-sharpe_ann / (dsr_std * np.sqrt(2)))))))

        st.markdown("##### 🛡️ Métricas Diarias de Riesgo Extremo & Performance")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Daily VaR 99% ($ USD)", f"${var_99_usd:,.2f}", f"-{var_99_pct*100:.2f}%")
        m_col2.metric("Expected Shortfall CVaR 99%", f"${cvar_99_usd:,.2f}", f"-{cvar_99_pct*100:.2f}%")
        m_col3.metric("Annualized Sharpe Ratio", f"{sharpe_ann:.2f}")
        m_col4.metric("Deflated Sharpe (DSR Prob)", f"{dsr_prob*100:.1f}%", help="Probabilidad de que el Sharpe no sea fruto de overfitting.")

        st.markdown("---")
        st.markdown("##### 🎲 Simulación de Monte Carlo (1,000 Caminos a 1 Año - 252 Días)")

        n_sims = 1000
        n_days = 252
        
        sim_paths = np.zeros((n_days, n_sims))
        sim_paths[0] = capital

        for t in range(1, n_days):
            random_returns = np.random.choice(port_returns, size=n_sims, replace=True)
            sim_paths[t] = sim_paths[t-1] * (1 + random_returns)

        percentile_5 = np.percentile(sim_paths, 5, axis=1)
        percentile_50 = np.percentile(sim_paths, 50, axis=1)
        percentile_95 = np.percentile(sim_paths, 95, axis=1)

        fig_mc = go.Figure()
        for s in range(30):
            fig_mc.add_trace(go.Scatter(y=sim_paths[:, s], mode='lines', line=dict(color='rgba(212, 175, 55, 0.08)', width=1), showlegend=False))

        fig_mc.add_trace(go.Scatter(y=percentile_95, mode='lines', name='P95 (Escenario Optimista)', line=dict(color='#22c55e', width=2)))
        fig_mc.add_trace(go.Scatter(y=percentile_50, mode='lines', name='Mediana P50 (Esperado)', line=dict(color='#d4af37', width=2.5)))
        fig_mc.add_trace(go.Scatter(y=percentile_5, mode='lines', name='P5 (Escenario Adverso)', line=dict(color='#ef4444', width=2)))

        fig_mc.update_layout(
            title="Distribución Estocástica de Capital (Monte Carlo 1,000 Caminos)",
            xaxis_title="Días de Trading (1 Año)", yaxis_title="Capital ($ USD)",
            template="plotly_dark", paper_bgcolor="#08080a", plot_bgcolor="#0d0d11",
            height=400, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🗺️ Matriz de Correlación Cruzada de Activos")

        corr_matrix = df_returns.corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale=[[0, "#08080a"], [0.5, "#3f3f46"], [1, "#d4af37"]],
            aspect="auto", title="Correlaciones Cruzadas de la Cartera"
        )
        fig_corr.update_layout(
            template="plotly_dark", paper_bgcolor="#08080a", plot_bgcolor="#0d0d11",
            height=500, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No fue posible cargar datos de retornos para la simulación de riesgo.")
    
