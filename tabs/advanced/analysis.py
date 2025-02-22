import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def perform_analysis(sales_series, indices_data, debug: bool = False):
    st.write("Before processing:")
    st.write(f"Sales series: {len(sales_series)} points, date range: {sales_series.index.min()} to {sales_series.index.max()}")
    st.write(f"Indices data: {indices_data.shape[1]} series, date range: {indices_data.index.min()} to {indices_data.index.max()}")

    # Ensure sales_series has a name
    if sales_series.name is None:
        sales_series.name = 'Sales'
    
    # Align indices to sales dates
    common_dates = sales_series.index.intersection(indices_data.index)
    st.write(f"Common dates: {len(common_dates)} points")
    if debug:
        st.write("Common dates:", common_dates)
    
    if len(common_dates) == 0:
        st.error("No overlapping dates between sales and indices data")
        return {
            'data': pd.DataFrame(),
            'combined_scores': pd.DataFrame(),
            'top_20_indices': pd.DataFrame()
        }
    
    # Restrict both datasets to common dates
    sales_series = sales_series[common_dates]
    indices_data = indices_data.loc[common_dates]
    if debug:
        st.write("Sales series after restricting to common dates:")
        st.table(sales_series.head())
        st.write("Indices data after restricting to common dates:")
        st.table(indices_data.head())
    
    # Ensure indices data is numeric
    indices_data = indices_data.apply(pd.to_numeric, errors='coerce')
    initial_columns = indices_data.columns.tolist()

    # Drop columns with all NaN values
    indices_data = indices_data.dropna(axis=1, how='all')
    dropped_nan_cols = set(initial_columns) - set(indices_data.columns)
    if dropped_nan_cols:
        with st.expander("Dropped indices with all NaN values"):
            st.warning(f"Dropped indices: {dropped_nan_cols}")
    if debug:
        st.write("Indices data after dropping all-NaN columns:")
        st.table(indices_data.head())

    # Drop columns with constant values
    constant_cols = [col for col in indices_data.columns if indices_data[col].nunique() <= 1]
    if constant_cols:
        indices_data = indices_data.drop(columns=constant_cols)
        with st.expander("Dropped constant indices"):
            st.warning(f"Dropped constant indices: {constant_cols}")
    if debug:
        st.write("Indices data after dropping constant columns:")
        st.table(indices_data.head())

    # Combine data
    data = pd.concat([sales_series, indices_data], axis=1)
    if debug:
        st.write("Combined data before cleaning:")
        st.table(data.head())
    
    st.write("After alignment:")
    st.write(f"Combined shape: {data.shape}")
    st.write(f"Date range: {data.index.min()} to {data.index.max()}")
    st.write(f"Missing values: {data.isnull().sum().sum()}")

    data = data.dropna()
    if debug:
        st.write("Data after dropping NA:")
        st.table(data.head())
    st.write(f"Data shape after cleaning: {data.shape}")

    if data.empty or data.shape[1] < 2:
        st.error("No valid data available after cleaning indices.")
        return {
            'data': pd.DataFrame(),
            'combined_scores': pd.DataFrame(),
            'top_20_indices': pd.DataFrame()
        }

    correlations = data.corr()[sales_series.name].drop(sales_series.name).sort_values(ascending=False)
    correlations = correlations.dropna().sort_values(ascending=False)

    if correlations.empty:
        st.warning("No valid correlations could be calculated.")
        return {
            'data': data,
            'combined_scores': pd.DataFrame(),
            'top_20_indices': pd.DataFrame()
        }

    try:
        with st.spinner('Performing Granger causality tests...'):
            granger_p_values = {}
            progress_bar = st.progress(0)
            total = len(indices_data.columns)
            errors = []
                
            for idx, column in enumerate(indices_data.columns):
                try:
                    if len(data) > 4:
                        test_result = grangercausalitytests(data[[sales_series.name, column]], maxlag=4, verbose=False)
                        p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(4)]
                        granger_p_values[column] = min(p_values)
                    else:
                        granger_p_values[column] = float('nan')
                except Exception as e:
                    granger_p_values[column] = float('nan')
                    errors.append(f"Granger causality test failed for {column}: {e}")
                progress_bar.progress((idx + 1) / total)
                
            progress_bar.empty()
            granger_results = pd.Series(granger_p_values).dropna()

            if errors:
                with st.expander("Granger Causality Test Errors"):
                    for error in errors:
                        st.warning(error)

        with st.spinner('Computing feature importance...'):
            model = RandomForestRegressor()
            model.fit(indices_data.fillna(0), sales_series.loc[indices_data.index])
            importances = pd.Series(model.feature_importances_, index=indices_data.columns)
            importances = importances.sort_values(ascending=False)

        combined_scores = pd.DataFrame({
            'Correlation': correlations,
            'Granger Causality p-value': granger_results,
            'Feature Importance': importances
        }).dropna()

        if combined_scores.empty:
            st.warning("No valid combined scores could be calculated.")
            return {
                'data': data,
                'combined_scores': pd.DataFrame(),
                'top_20_indices': pd.DataFrame()
            }

        rank_cols = []
        for col in combined_scores.columns:
            rank_name = f'{col}_rank'
            if col == 'Granger Causality p-value':
                combined_scores[rank_name] = combined_scores[col].rank()
            else:
                combined_scores[rank_name] = combined_scores[col].rank(ascending=False)
            rank_cols.append(rank_name)

        combined_scores['Total Score'] = combined_scores[rank_cols].mean(axis=1)
        top_20_indices = combined_scores.nsmallest(20, 'Total Score')

        return {
            'data': data,
            'combined_scores': combined_scores,
            'top_20_indices': top_20_indices
        }
    except Exception as e:
        with st.expander("Analysis Error Details"):
            st.error(f"An error occurred during analysis: {e}")
        return {
            'data': pd.DataFrame(),
            'combined_scores': pd.DataFrame(),
            'top_20_indices': pd.DataFrame()
        }
