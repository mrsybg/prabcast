import streamlit as st

def get_user_inputs(min_date, max_date):
    st.subheader('Konfiguration der Datenanreicherung')
    
    # ── Zeitraum ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Startdatum', min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input('Enddatum', max_date, min_value=min_date, max_value=max_date)
    
    # ── Externe Indizes ──────────────────────────────────────
    with st.expander("Externe Indexquellen konfigurieren", expanded=True):
        col_ind, col_loc = st.columns(2)
        with col_ind:
            industries = st.multiselect(
                'Industriezweige (GICS)', 
                [
                    'Energy', 'Materials', 'Industrials',
                    'Consumer Discretionary', 'Consumer Staples', 'Health Care',
                    'Financials', 'Information Technology', 'Communication Services',
                    'Utilities', 'Real Estate'
                ],
                help='GICS-Sektoren für relevante Industrieindizes'
            )
        with col_loc:
            locations = st.multiselect(
                'Regionen / Märkte', 
                [
                    'North America', 'Europe', 'Asia Pacific', 'Latin America',
                    'USA', 'Germany', 'France', 'UK',
                    'China', 'Japan', 'India', 'Brazil'
                ],
                help='Regionale Marktindizes'
            )
        
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            fmp_stocks_input = st.text_input(
                'Aktien / ETF Ticker',
                placeholder='z. B. AAPL, MSFT',
                help='Yahoo-Finance-Ticker, kommagetrennt'
            )
        with col_t2:
            fmp_commodities_input = st.text_input(
                'Rohstoff-Futures',
                placeholder='z. B. CL=F, NG=F',
                help='Yahoo-Finance-Futures-Ticker (enden auf =F)'
            )
        with col_t3:
            metal_prices_input = st.text_input(
                'Metallpreise',
                placeholder='z. B. GC=F, SI=F',
                help='Gold (GC=F), Silber (SI=F) etc.'
            )

    # ── Umfang ────────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        series_per_category = st.number_input(
            'Zeitreihen pro Kategorie',
            min_value=1, max_value=100, value=50,
            help='Wie viele Index-Zeitreihen pro Industrie-/Regionsgruppe abgerufen werden'
        )
    with col_s2:
        number_of_sources = st.number_input(
            'Genutzte Datenquellen insgesamt',
            min_value=1, max_value=50, value=10,
            help='Maximale Anzahl der Top-Indizes, die in der Analyse verwendet werden'
        )
    
    return (
        industries,
        locations,
        [t.strip() for t in fmp_stocks_input.split(',') if t.strip()],
        [t.strip() for t in fmp_commodities_input.split(',') if t.strip()],
        [t.strip() for t in metal_prices_input.split(',') if t.strip()],
        series_per_category,
        number_of_sources,
        start_date,
        end_date
    )
