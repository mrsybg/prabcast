import pandas as pd
from fredapi import Fred
import requests
import time
from typing import List, Dict, Tuple
import streamlit as st
from datetime import timedelta
import yfinance as yf

pd.set_option('future.no_silent_downcasting', True)

def fetch_with_retry(func, max_retries=3, delay=1):
    """Helper function to retry failed API calls"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise e

def fetch_yfinance_data(ticker: str, date_range: pd.DatetimeIndex, debug: bool = False) -> pd.Series:
    """Unified function to fetch any financial data from Yahoo Finance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(
            start=date_range.min(),
            end=date_range.max() + timedelta(days=1),
            interval='1mo'  # Monthly data
        )
        
        if debug:
            with st.sidebar.expander(f"Raw Yahoo Finance Data für {ticker}"):
                st.write(df)
                
        if df.empty:
            st.warning(f"Keine Daten für {ticker} gefunden")
            return pd.Series(index=date_range, name=ticker)
        
        # Use Close price for consistency
        series = df['Close']
        series.name = ticker
        # Normalize index to remove time components
        series.index = series.index.normalize()
        # Reindex to match the provided date_range
        series = series.reindex(date_range).ffill().bfill()
        
        if debug:
            with st.sidebar.expander(f"Processed Series für {ticker}"):
                st.write(series.head())
                
        return series
    except Exception as e:
        st.warning(f"Yahoo Finance Datenabruf für {ticker} fehlgeschlagen: {str(e)}")
        return pd.Series(index=date_range, name=ticker)

def fetch_indices_data(
    industries: List[str],
    locations: List[str],
    date_range: pd.DatetimeIndex,
    series_per_category: int,
    test_mode: bool = False,
    stocks: List[str] = [],
    commodities: List[str] = [],
    metals: List[str] = [],
    number_of_sources: int = 0
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    all_indices_data = []
    index_names = {}

    fred = Fred(api_key='96b1249e3a1e6babd8d2235a8a87ab48')

    # Updated category mappings
    category_mappings = {
        'industries': {
            'Energy': {'id': 32258, 'name': 'Energy'},
            'Materials': {'id': 32259, 'name': 'Materials'},
            'Industrials': {'id': 32273, 'name': 'Industrials'},
            'Consumer Discretionary': {'id': 32260, 'name': 'Consumer Discretionary'},
            'Consumer Staples': {'id': 32261, 'name': 'Consumer Staples'},
            'Health Care': {'id': 32257, 'name': 'Health Care'},
            'Financials': {'id': 32256, 'name': 'Financials'},
            'Information Technology': {'id': 32455, 'name': 'Information Technology'},
            'Communication Services': {'id': 32262, 'name': 'Communication Services'},
            'Utilities': {'id': 32263, 'name': 'Utilities'},
            'Real Estate': {'id': 32264, 'name': 'Real Estate'}
        },
        'locations': {
            'North America': {'id': 32451, 'name': 'North American Markets'},
            'Europe': {'id': 32254, 'name': 'European Markets'},
            'Asia Pacific': {'id': 32253, 'name': 'Asian Pacific Markets'},
            'Latin America': {'id': 32265, 'name': 'Latin American Markets'},
            'USA': {'id': 32454, 'name': 'USA'},
            'Germany': {'id': 32452, 'name': 'Germany'},
            'France': {'id': 32453, 'name': 'France'},
            'UK': {'id': 32266, 'name': 'United Kingdom'},
            'China': {'id': 32267, 'name': 'China'},
            'Japan': {'id': 32268, 'name': 'Japan'},
            'India': {'id': 32269, 'name': 'India'},
            'Brazil': {'id': 32270, 'name': 'Brazil'}
        }
    }

    def process_category(
        category_id: int,
        category_name: str,
        series_per_category: int,
        test_mode: bool,
        date_range: pd.DatetimeIndex,
        fred: Fred,
        index_names: Dict[str, str]
    ) -> pd.DataFrame:
        series_list = fetch_category_data(category_id, series_per_category, test_mode)
        category_series = []
        errors = []

        with st.spinner(f'Processing {category_name} data...'):
            progress_bar = st.progress(0)
            for idx, series in enumerate(series_list):
                series_id = series['id']
                index_names[series_id] = series['title']

                try:
                    series_data = fred.get_series(
                        series_id,
                        observation_start=date_range.min(),
                        observation_end=date_range.max()
                    )

                    series_data.index = pd.to_datetime(series_data.index, errors='coerce')
                    series_data = series_data.dropna()
                    series_data = pd.to_numeric(series_data, errors='coerce')

                    # Resample auf Monatsende
                    series_data = series_data.resample('M').last()
                    series_data = series_data.reindex(date_range)
                    series_data = series_data.ffill().bfill()

                    if series_data.isnull().all():
                        errors.append(f"Series {series_id} enthält keine numerischen Daten und wird übersprungen.")
                        continue

                    processed_series = pd.Series(
                        series_data.values,
                        index=date_range,
                        name=series_id
                    )
                    category_series.append(processed_series)
                except Exception as e:
                    errors.append(f"Fehler bei der Verarbeitung von {series_id}: {e}")

                progress_bar.progress((idx + 1) / len(series_list))
                time.sleep(0.5)
            progress_bar.empty()

        if category_series:
            st.write(f"Processed {len(category_series)} von {len(series_list)} Series für {category_name}")
            if errors:
                with st.expander(f"Fehler in {category_name}"):
                    for error in errors:
                        st.warning(error)
            return pd.concat(category_series, axis=1)
        else:
            with st.expander(f"Keine gültigen Daten für {category_name}"):
                for error in errors:
                    st.warning(error)
            return pd.DataFrame(index=date_range)

    for category_type, selections in [
        ('industries', industries),
        ('locations', locations)
    ]:
        for selection in selections:
            if selection in category_mappings[category_type]:
                cat_info = category_mappings[category_type][selection]
                df = process_category(
                    cat_info['id'],
                    cat_info['name'],
                    series_per_category,
                    test_mode,
                    date_range,
                    fred,
                    index_names
                )
                if not df.empty:
                    all_indices_data.append(df)
                    st.write(f"Processed {cat_info['name']}: {df.shape[1]} Series")
            else:
                st.warning(f"{selection} nicht in {category_type} mappings gefunden")

    # Process Yahoo Finance data
    if stocks:
        with st.spinner('Fetching stock data...'):
            stock_data = pd.DataFrame()
            for ticker in stocks:
                series = fetch_yfinance_data(ticker, date_range)
                stock_data[series.name] = series
                index_names[series.name] = f"Stock: {ticker}"
                st.success(f"Fetched stock data for {ticker}")
                time.sleep(1)
            if not stock_data.empty:
                all_indices_data.append(stock_data)
                st.write(f"Added {stock_data.shape[1]} stock sources")
    
    if commodities:
        with st.spinner('Fetching commodity data...'):
            comm_data = pd.DataFrame()
            for ticker in commodities:
                # Add common commodity suffixes if not present
                if not ticker.endswith(('=F', '-USD')):
                    ticker = f"{ticker}=F"
                series = fetch_yfinance_data(ticker, date_range)
                comm_data[series.name] = series
                index_names[series.name] = f"Commodity: {ticker}"
                st.success(f"Fetched commodity data for {ticker}")
                time.sleep(1)
            if not comm_data.empty:
                all_indices_data.append(comm_data)
                st.write(f"Added {comm_data.shape[1]} commodity sources")
    
    if metals:
        with st.spinner('Fetching metal price data...'):
            metal_data = pd.DataFrame()
            metal_mapping = {
                'GOLD': 'GC=F',
                'SILVER': 'SI=F',
                'PLATINUM': 'PL=F',
                'PALLADIUM': 'PA=F',
                'COPPER': 'HG=F'
            }
            for ticker in metals:
                yf_symbol = metal_mapping.get(ticker.upper(), ticker)
                series = fetch_yfinance_data(yf_symbol, date_range)
                metal_data[ticker.upper()] = series
                index_names[ticker.upper()] = f"Metal: {ticker}"
                st.success(f"Fetched metal price data for {ticker}")
                time.sleep(1)
            if not metal_data.empty:
                all_indices_data.append(metal_data)
                st.write(f"Added {metal_data.shape[1]} metal price sources")
    
    if all_indices_data:
        final_data = pd.concat(all_indices_data, axis=1)
        if number_of_sources > 0 and final_data.shape[1] > number_of_sources:
            final_data = final_data.iloc[:, :number_of_sources]
            index_names = {k: index_names[k] for k in list(final_data.columns)}
        st.write(f"Final shape: {final_data.shape}")
        return final_data, index_names, number_of_sources
    else:
        return pd.DataFrame(index=date_range), {}, number_of_sources

def fetch_category_data(
    category_id: int,
    required_series: int,
    test_mode: bool = False
) -> List[Dict[str, str]]:
    all_series = []
    offset = 0
    base_url = "https://api.stlouisfed.org/fred/category/series"
    api_key = "96b1249e3a1e6babd8d2235a8a87ab48"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "limit": 1000
    }
    with st.spinner(f'Fetching category {category_id} data...'):
        while len(all_series) < required_series:
            params["category_id"] = category_id
            params["offset"] = offset

            def make_request():
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                return response.json()

            data = fetch_with_retry(make_request)
            if "seriess" not in data or not data["seriess"]:
                break

            series_info = [
                {'id': series['id'], 'title': series['title']}
                for series in data['seriess']
            ]
            all_series.extend(series_info)
            if len(series_info) < 1000:
                break
            offset += 1000
            time.sleep(0.5)
            if test_mode and len(all_series) >= required_series // 2:
                st.info(f'Test mode: Limited to {required_series // 2} series')
                break
    return all_series[:required_series]
