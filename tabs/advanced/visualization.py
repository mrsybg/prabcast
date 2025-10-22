# visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from setup_module.helpers import *

def make_unique_columns(df):
    """Append suffixes to duplicate column names to make them unique."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def display_results(results, num_top_indices):
    if not results or 'top_20_indices' not in results:
        st.error("Keine Ergebnisse.")
        return
    
    st.header('Prognoseergebnisse')
    index_names = results.get('index_names', {})
    
    # Make index_names unique first to prevent duplicates from the start
    unique_index_names = {}
    name_counts = {}
    for key, name in index_names.items():
        if name not in name_counts:
            name_counts[name] = 0
            unique_index_names[key] = name
        else:
            name_counts[name] += 1
            unique_index_names[key] = f"{name}_{name_counts[name]}"
    
    index_names = unique_index_names

    # Display top 5 indices based on total score (keeping only this table)
    st.subheader('Top 5 Datensätze basierend auf dem Gesamtscore')
    top_5_indices = results['top_20_indices'].head(5)
    top_5_indices_styled = format_summary_table(top_5_indices, top_5_indices.columns[1:8].tolist(), decimal_places=4)
    st.table(top_5_indices_styled)

    # Correlation plot
    with st.expander("Korrelation mit Absatz"):
        try:
            correlations = results['combined_scores']['Correlation'].rename(index=index_names)
            fig = px.bar(correlations, title='Korrelation mit Absatz')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error displaying correlation plot: {e}")

    # Feature importance plot
    with st.expander("Feature Importance"):
        try:
            importances = results['combined_scores']['Feature Importance'].rename(index=index_names)
            fig = px.bar(importances, title='Feature Importance')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error displaying feature importance plot: {e}")

    # Granger causality p-values plot
    with st.expander("Granger Causality P-Werte"):
        try:
            granger_p_values = results['combined_scores']['Granger Causality p-value'].rename(index=index_names)
            fig = px.bar(granger_p_values, title='Granger Causality P-Werte')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error displaying Granger causality plot: {e}")

    # Sales time series with top indices - only use top N indices
    st.subheader('Absatzzeitreihe und Top Datensätze nach Datenanreicherung')
    top_indices = results['top_20_indices'].head(num_top_indices).index.tolist()  # Only take top N indices
    final_df = results['data'][[results['data'].columns[0]] + top_indices].rename(columns=index_names)
    # Make column names unique immediately to avoid DuplicateError in Plotly / narwhals
    final_df = make_unique_columns(final_df)
    
    fig = go.Figure()
    for col in final_df.columns:
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df[col], mode='lines', name=col))
    fig.update_layout(title='Absatzzeitreihe und Top Datensätze nach Datenanreicherung', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)
    
    # Normalized sales time series with top indices
    st.subheader('Normalisiert: Absatzzeitreihe und Top Datensätze nach Datenanreicherung')
    normalized_df = final_df.apply(lambda x: (x - x.mean()) / x.std())
    
    fig = go.Figure()
    for col in normalized_df.columns:
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df[col], mode='lines', name=col))
    fig.update_layout(title='Normalisiert: Absatzzeitreihe und Top Datensätze nach Datenanreicherung', xaxis_title='Date', yaxis_title='Normalized Value')
    st.plotly_chart(fig)

    # Final DataFrame - now only includes top N indices
    st.subheader('Datensatz zur Weiterverarbeitung')
    # ensure we reuse the already-unique final_df (or recreate and deduplicate again)
    final_df = results['data'][[results['data'].columns[0]] + top_indices].rename(columns=index_names)
    final_df = make_unique_columns(final_df)
    
    # format table (change decimal seperator)
    final_df = format_summary_table(final_df, final_df.columns[1:3].tolist())
    
    st.dataframe(final_df)