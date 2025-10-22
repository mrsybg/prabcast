import streamlit as st

def get_user_inputs(min_date, max_date):
    st.header('Fragebogen')
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Startdatum', min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input('Enddatum', max_date, min_value=min_date, max_value=max_date)
    
    # Industrieklassifikation (GICS Sektoren)
    industries = st.multiselect(
        'Wählen Sie relevante Industriezweige (GICS Sektoren)', 
        [
            'Energy',
            'Materials',
            'Industrials',
            'Consumer Discretionary',
            'Consumer Staples',
            'Health Care',
            'Financials',
            'Information Technology',
            'Communication Services',
            'Utilities',
            'Real Estate'
        ]
    )
    
    # Markt/Region
    locations = st.multiselect(
        'Wählen Sie relevante Regionen/Märkte', 
        [
            'North America',
            'Europe',
            'Asia Pacific',
            'Latin America',
            'USA',
            'Germany',
            'France',
            'UK',
            'China',
            'Japan',
            'India',
            'Brazil'
        ]
    )
    
    # Aktien/ETF as freitext (Yahoo Finance)
    fmp_stocks_input = st.text_input(
        'Aktien/ETF Ticker von Yahoo Finance (kommagetrennt, z.B. AAPL,MSFT)',
        help='Geben Sie die Yahoo Finance Ticker für Aktien/ETF ein'
    )
    
    # Rohstoffe (Yahoo Finance)
    fmp_commodities_input = st.text_input(
        'Rohstoff-Futures von Yahoo Finance (kommagetrennt, z.B. CL=F,NG=F)',
        help='Geben Sie die Yahoo Finance Ticker für Rohstoff-Futures ein. Futures enden mit =F'
    )
    
    # Metallpreise (Yahoo Finance)
    metal_prices_input = st.text_input(
        'Metallpreise von Yahoo Finance (kommagetrennt, z.B. GC=F,SI=F)',
        help='Geben Sie die Yahoo Finance Ticker für Metallpreise ein. Gold (GC=F), Silber (SI=F), etc.'
    )
    
    # Anzahl der Zeitreihen pro Kategorie (für Industrie/Region)
    series_per_category = st.number_input(
        'Anzahl der Zeitreihen pro Kategorie (Industrie/Region)',
        min_value=1,
        max_value=100,
        value=50
    )
    
    # Anzahl der genutzten Datenquellen insgesamt
    number_of_sources = st.number_input(
        'Anzahl der genutzten Datenquellen insgesamt',
        min_value=1,
        max_value=50,
        value=10
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
