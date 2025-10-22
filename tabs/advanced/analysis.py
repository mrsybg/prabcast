import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import numpy as np

def detect_quarterly_pattern(df):
    """
    Detect if dataframe has quarterly repeated values pattern and return columns with this pattern
    """
    quarterly_cols = []
    
    for col in df.columns:
        values = df[col].dropna().values
        if len(values) >= 12:  # Need at least a year of data
            is_quarterly = True
            # Check if values repeat every 3 months
            for i in range(len(values) // 3 - 1):
                if values[i*3] != values[(i+1)*3] or values[i*3+1] != values[(i+1)*3+1] or values[i*3+2] != values[(i+1)*3+2]:
                    is_quarterly = False
                    break
            if is_quarterly:
                quarterly_cols.append(col)
    
    return quarterly_cols

def interpolate_quarterly_data(df, quarterly_cols):
    """
    Apply cubic interpolation to columns with quarterly patterns
    """
    df_interpolated = df.copy()
    
    # First convert index to continuous numeric for better interpolation
    idx_numeric = np.arange(len(df))
    idx_map = {idx: num for idx, num in zip(df.index, idx_numeric)}
    
    for col in quarterly_cols:
        # Create temporary series with numeric index for interpolation
        temp_series = pd.Series(df[col].values, index=idx_numeric)
        
        # Find positions of the original quarterly values (every 3rd point)
        quarterly_positions = [i for i in range(0, len(temp_series), 3)]
        quarterly_values = temp_series.iloc[quarterly_positions]
        
        # Create a new series with just the quarterly points
        sparse_series = pd.Series(index=idx_numeric)
        sparse_series.iloc[quarterly_positions] = quarterly_values
        
        # Interpolate between these sparse points
        interpolated = sparse_series.interpolate(method='cubic')
        
        # Replace original column with interpolated values
        df_interpolated[col] = interpolated.values
    
    return df_interpolated

def perform_analysis(sales_series, indices_data, debug: bool = False):
    # ✨ DEBUG INFO in Expander
    with st.expander("🔍 Debug-Informationen", expanded=False):
        st.write(f"**Sales series:** {len(sales_series)} points, date range: {sales_series.index.min()} to {sales_series.index.max()}")
        st.write(f"**Indices data:** {indices_data.shape[1]} series, date range: {indices_data.index.min()} to {indices_data.index.max()}")

    if sales_series.name is None:
        sales_series.name = 'Sales'
    
    # Detect and fix quarterly patterns before proceeding
    quarterly_cols = detect_quarterly_pattern(indices_data)
    if quarterly_cols:
        with st.expander("Quarterly data detected", expanded=True):
            st.warning(f"Detected quarterly pattern in {len(quarterly_cols)} columns. Applying monthly interpolation.")
            st.write("Columns with quarterly pattern:")
            st.write(quarterly_cols[:10])  # Show first 10 for brevity
            if len(quarterly_cols) > 10:
                st.write(f"...and {len(quarterly_cols) - 10} more")
        
        # Apply cubic interpolation to quarterly columns
        indices_data = interpolate_quarterly_data(indices_data, quarterly_cols)
        
        # Show a sample of before/after
        with st.expander("Sample of interpolated data", expanded=False):
            if quarterly_cols:
                sample_col = quarterly_cols[0]
                st.write(f"Column: {sample_col}")
                
                # Create DataFrame to show before/after
                before_after = pd.DataFrame({
                    'Before': indices_data[sample_col],
                    'After': indices_data[sample_col]
                })
                
                st.write(before_after.head(12))
    
    common_dates = sales_series.index.intersection(indices_data.index)
    
    # ✨ DEBUG INFO in Expander
    with st.expander("🔍 Debug-Informationen", expanded=False):
        st.write(f"**Common dates found:** {len(common_dates)}")
        if debug:
            st.write("Common dates:", common_dates)

    if len(common_dates) == 0:
        st.error("No overlapping dates between sales and indices data")
        return {
            'data': pd.DataFrame(),
            'combined_scores': pd.DataFrame(),
            'top_20_indices': pd.DataFrame()
        }

    sales_series = sales_series[common_dates]
    indices_data = indices_data.loc[common_dates]
    
    # ✨ DEBUG INFO in Expander
    if debug:
        with st.expander("🔍 Debug-Informationen", expanded=False):
            st.write("**Sales series after restricting to common dates:**")
            st.dataframe(sales_series.head())
            st.write("**Indices data after restricting to common dates:**")
            st.dataframe(indices_data.head())

    indices_data = indices_data.apply(pd.to_numeric, errors='coerce')
    initial_columns = indices_data.columns.tolist()

    indices_data = indices_data.dropna(axis=1, how='all')
    dropped_nan_cols = set(initial_columns) - set(indices_data.columns)
    if dropped_nan_cols:
        with st.expander("Dropped indices with all NaN values"):
            st.warning(f"Dropped indices: {dropped_nan_cols}")
    
    # ✨ DEBUG INFO in Expander
    if debug:
        with st.expander("🔍 Debug-Informationen", expanded=False):
            st.write("**Indices data after dropna (all NaN):**")
            st.dataframe(indices_data.head())

    constant_cols = [col for col in indices_data.columns if indices_data[col].nunique() <= 1]
    if constant_cols:
        indices_data = indices_data.drop(columns=constant_cols)
        with st.expander("Dropped constant indices"):
            st.warning(f"Dropped constant indices: {constant_cols}")
    
    # ✨ DEBUG INFO in Expander
    if debug:
        with st.expander("🔍 Debug-Informationen", expanded=False):
            st.write("**Indices data after dropping constant columns:**")
            st.dataframe(indices_data.head())

    data = pd.concat([sales_series, indices_data], axis=1)
    
    # ✨ DEBUG INFO in Expander
    with st.expander("🔍 Debug-Informationen", expanded=False):
        if debug:
            st.write("**Combined data before cleaning:**")
            st.dataframe(data.head())
        st.write(f"**Combined data shape before cleaning:** {data.shape}")
        st.write(f"**Total missing values before cleaning:** {data.isnull().sum().sum()}")

    data = data.dropna()
    
    # ✨ DEBUG INFO in Expander
    with st.expander("🔍 Debug-Informationen", expanded=False):
        if debug:
            st.write("**Data after dropping NA:**")
            st.dataframe(data.head())
        st.write(f"**Data shape after cleaning:** {data.shape}")

    if data.empty or data.shape[1] < 2:
        st.error("No valid data available after cleaning indices.")
        return {
            'data': pd.DataFrame(),
            'combined_scores': pd.DataFrame(),
            'top_20_indices': pd.DataFrame()
        }

    correlations = data.corr()[sales_series.name].drop(sales_series.name).dropna().sort_values(ascending=False)
    
    # ✨ DEBUG INFO in Expander
    with st.expander("🔍 Debug-Informationen", expanded=False):
        st.write("**Correlations computed (Top 5):**")
        st.dataframe(correlations.head())

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
            if debug:
                st.write("[DEBUG] Feature importances computed:", importances.head())

        combined_scores = pd.DataFrame({
            'Correlation': correlations,
            'Granger Causality p-value': granger_results,
            'Feature Importance': importances
        }).dropna()
        if debug:
            st.write("[DEBUG] Combined scores before ranking:", combined_scores.head())

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

        if debug:
            st.write("[DEBUG] Combined scores with total score added:", combined_scores.head())
            st.write("[DEBUG] Top 20 indices based on total score:", top_20_indices.head())

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
