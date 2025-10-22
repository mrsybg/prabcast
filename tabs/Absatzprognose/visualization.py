# visualization.py
import os
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font = dict(family='Arial')

def create_product_dashboard(train_data, test_data, forecasts, metrics, product, results_dir='results'):
    """
    Erstellt ein interaktives HTML-Dashboard für ein einzelnes Produkt.
    """
    product_dir = os.path.join(results_dir, 'products', product)
    os.makedirs(product_dir, exist_ok=True)
    
    # Erstellen der Vorhersage-Figur
    fig1 = go.Figure()
    
    # Trainingsdaten hinzufügen
    fig1.add_trace(go.Scatter(x=train_data.index, y=train_data[product], 
                            name='Training Data', line=dict(color='blue')))
    
    # Testdaten hinzufügen
    fig1.add_trace(go.Scatter(x=test_data.index, y=test_data[product], 
                            name='Test Data', line=dict(color='green')))
    
    # Vorhersagen der verschiedenen Modelle hinzufügen
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'olive', 
             'teal', 'gold', 'darkblue', 'crimson']
    
    for (model_name, model_forecasts), color in zip(forecasts.items(), colors):
        if model_name == 'Ensemble':
            continue  # Ensemble wird separat behandelt
        fig1.add_trace(go.Scatter(x=model_forecasts.index, y=model_forecasts[product],
                                name=f'Forecast {model_name}', 
                                line=dict(color=color, dash='dash')))
    
    fig1.update_layout(
        title=f"Forecasts for {product}",
        xaxis_title="Date",
        yaxis_title="Value",
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100, b=150)
    )
    
    # Erstellen der Tabelle mit Metriken
    table_data = []
    headers = ['Model', 'MAE', 'RMSE', 'sMAPE', 'Bias', 'Theils_U']  # Removed 'Time (s)', 'Memory (MB)'
    for model_name, model_metrics in metrics.items():
        row = [model_name] + [f"{model_metrics.loc[product, metric]:.6f}" 
               for metric in ['MAE', 'RMSE', 'sMAPE', 'Bias', 'Theils_U']] + \
              [f"{model_metrics.loc[product, 'Time']:.6f}" if not pd.isna(model_metrics.loc[product, 'Time']) else 'N/A',
               f"{model_metrics.loc[product, 'Memory']:.6f}" if not pd.isna(model_metrics.loc[product, 'Memory']) else 'N/A']
        table_data.append(row)
    
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=headers,
                   fill_color='paleturquoise',
                   align='center',
                   font=dict(size=14)),
        cells=dict(values=list(zip(*table_data)),
                  fill_color='lavender',
                  align='center',
                  font=dict(size=12),
                  height=30)
    )])
    
    fig2.update_layout(
        title="Model Performance Metrics",
        height=300
    )
    
    # Ensemble-Vorhersage hinzufügen, wenn vorhanden
    if 'Ensemble' in forecasts:
        fig1.add_trace(go.Scatter(x=forecasts['Ensemble'].index, y=forecasts['Ensemble'][product],
                                name='Forecast Ensemble', 
                                line=dict(color='black', dash='dot')))
    
    # Generieren des HTML-Dashboards
    with open(os.path.join(product_dir, f'{product}_dashboard.html'), 'w') as f:
        f.write(f"""
        <html>
            <head><title>{product} Dashboard</title></head>
            <body>
                <div>{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                <div>{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            </body>
        </html>
        """)
    
    # Speichern der Metriken als CSV
    pd.DataFrame(table_data, columns=headers).to_csv(
        os.path.join(product_dir, f'{product}_metrics.csv'), index=False)

def create_horizon_dashboard(metrics, horizon, results_dir='results'):
    horizon_dir = os.path.join(results_dir, f'{horizon}_months')
    os.makedirs(horizon_dir, exist_ok=True)

    figs = []
    
    # 1. Average sMAPE per Model
    avg_smape = pd.DataFrame({model: df['sMAPE'].mean() for model, df in metrics.items()}, index=['Average sMAPE']).T
    fig_smape = go.Figure(data=[
        go.Bar(x=avg_smape.index, y=avg_smape['Average sMAPE'])
    ])
    fig_smape.update_layout(title=f'Average sMAPE per Model - {horizon} Month Horizon',
                         xaxis_title="Model", yaxis_title="Average sMAPE (%)")
    figs.append(fig_smape)

    # 2. Model Ranking (how often each model was best)
    best_counts = pd.Series(dtype=str)
    for product in metrics[next(iter(metrics))].index:
        model_scores = {model: metrics[model].loc[product, 'sMAPE'] for model in metrics.keys()}
        best_model = min(model_scores.items(), key=lambda x: x[1])[0]
        best_counts = best_counts.append(pd.Series([best_model]))
    
    ranking = best_counts.value_counts()
    fig_rank = go.Figure(data=[
        go.Bar(x=ranking.index, y=ranking.values)
    ])
    fig_rank.update_layout(title=f'Best Model Frequency - {horizon} Month Horizon',
                          xaxis_title="Model", yaxis_title="Times Best")
    figs.append(fig_rank)

    # 3. Performance Stability (sMAPE variance)
    smape_std = pd.DataFrame({model: df['sMAPE'].std() for model, df in metrics.items()}, index=['sMAPE Std']).T
    fig_stability = go.Figure(data=[
        go.Bar(x=smape_std.index, y=smape_std['sMAPE Std'])
    ])
    fig_stability.update_layout(title=f'Model Stability (sMAPE Standard Deviation) - {horizon} Month Horizon',
                              xaxis_title="Model", yaxis_title="sMAPE Std (%)")
    figs.append(fig_stability)

    # 4. Memory vs Average sMAPE Trade-off
    avg_metrics = {model: {'sMAPE': metrics[model]['sMAPE'].mean(), 
                          'Memory': metrics[model]['Memory'].mean()} 
                  for model in metrics.keys()}
    
    fig_memory = go.Figure(data=[
        go.Bar(x=list(avg_metrics.keys()), y=[avgs['Memory'] for avgs in avg_metrics.values()])
    ])
    fig_memory.update_layout(title=f'Average Memory Usage per Model - {horizon} Month Horizon',
                             xaxis_title="Model", yaxis_title="Memory (MB)")
    figs.append(fig_memory)

    # 5. Time vs Average sMAPE Trade-off
    avg_metrics_time = {model: {'sMAPE': metrics[model]['sMAPE'].mean(), 
                                'Time': metrics[model]['Time'].mean()} 
                        for model in metrics.keys()}
    
    fig_time = go.Figure(data=[
        go.Bar(x=list(avg_metrics_time.keys()), y=[avgs['Time'] for avgs in avg_metrics_time.values()])
    ])
    fig_time.update_layout(title=f'Average Computation Time per Model - {horizon} Month Horizon',
                           xaxis_title="Model", yaxis_title="Time (s)")
    figs.append(fig_time)

    # 6. Boxplot of Errors
    error_data = []
    labels = []
    for model in metrics.keys():
        error_data.append(metrics[model]['sMAPE'])
        labels.extend([model] * len(metrics[model]['sMAPE']))
    
    fig_box = go.Figure(data=[go.Box(x=labels, y=error_data)])
    fig_box.update_layout(title=f'sMAPE Distribution per Model - {horizon} Month Horizon',
                         xaxis_title="Model", yaxis_title="sMAPE (%)")
    figs.append(fig_box)

    # 7. Success Rate (% of forecasts within 20% error)
    success_rates = {}
    threshold = 0.20  # 20% error threshold
    for model in metrics.keys():
        success = (metrics[model]['sMAPE'] < threshold * 100).mean() * 100  # Convert to percentage
        success_rates[model] = success
    
    fig_success = go.Figure(data=[
        go.Bar(x=list(success_rates.keys()), y=list(success_rates.values()))
    ])
    fig_success.update_layout(title=f'Success Rate (Within {threshold*100}% Error) - {horizon} Month Horizon',
                            xaxis_title="Model", yaxis_title="Success Rate (%)")
    figs.append(fig_success)

    # Create combined dashboard
    dashboard_html = "<html><head><title>Horizon Dashboard</title></head><body>"
    for fig in figs:
        dashboard_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    dashboard_html += "</body></html>"

    with open(os.path.join(horizon_dir, 'horizon_dashboard.html'), 'w') as f:
        f.write(dashboard_html)

def create_total_dashboard(all_horizons_metrics, forecast_horizons, results_dir='results'):
    """
    Erstellt ein umfassendes Dashboard über alle Horizonte.
    """
    total_dir = os.path.join(results_dir, 'total')
    os.makedirs(total_dir, exist_ok=True)
    figs = []
    
    # 1. Performance Evolution Across Horizons
    metric_evolution = {model: {horizon: metrics['sMAPE'].mean() 
                              for horizon, metrics_dict in all_horizons_metrics.items()
                              for m_name, metrics in metrics_dict.items() 
                              if m_name == model}
                       for model in all_horizons_metrics[forecast_horizons[0]].keys()}
    
    fig_evolution = go.Figure()
    for model, values in metric_evolution.items():
        fig_evolution.add_trace(go.Scatter(
            x=list(values.keys()), y=list(values.values()),
            mode='lines+markers', name=model
        ))
    fig_evolution.update_layout(title='sMAPE Evolution Across Horizons',
                              xaxis_title='Forecast Horizon (months)',
                              yaxis_title='Average sMAPE (%)')
    figs.append(fig_evolution)

    # 2. Model Consistency (Performance Stability Across Horizons)
    consistency = {model: np.std([metrics['sMAPE'].mean() 
                                for metrics_dict in all_horizons_metrics.values()
                                for m_name, metrics in metrics_dict.items() 
                                if m_name == model])
                  for model in all_horizons_metrics[forecast_horizons[0]].keys()}
    
    fig_consistency = go.Figure(data=[
        go.Bar(x=list(consistency.keys()), y=list(consistency.values()))
    ])
    fig_consistency.update_layout(title='Model Consistency (Lower is Better)',
                                xaxis_title='Model',
                                yaxis_title='sMAPE Standard Deviation Across Horizons')
    figs.append(fig_consistency)

    # 3. Overall Resource Efficiency
    resource_efficiency = {model: np.mean([metrics['sMAPE'].mean() / (metrics['Time'].mean() + 1)
                                         for metrics_dict in all_horizons_metrics.values()
                                         for m_name, metrics in metrics_dict.items()
                                         if m_name == model and not pd.isna(metrics['Time'])])
                         for model in all_horizons_metrics[forecast_horizons[0]].keys()}
    
    fig_efficiency = go.Figure(data=[
        go.Bar(x=list(resource_efficiency.keys()), y=list(resource_efficiency.values()))
    ])
    fig_efficiency.update_layout(title='Overall Resource Efficiency (Lower is Better)',
                               xaxis_title='Model',
                               yaxis_title='sMAPE/Time Ratio')
    figs.append(fig_efficiency)

    # 4. Top 3 Models Summary
    top_models_html = "<div style='margin: 20px;'><h2>Top 3 Models per Horizon</h2>"
    for horizon in forecast_horizons:
        avg_performance = {model: metrics['sMAPE'].mean() 
                         for model, metrics in all_horizons_metrics[horizon].items()}
        top_3 = sorted(avg_performance.items(), key=lambda x: x[1])[:3]
        top_models_html += f"<h3>{horizon} Month Horizon:</h3><ul>"
        for model, smape in top_3:
            top_models_html += f"<li>{model}: sMAPE = {smape:.2f}%</li>"
        top_models_html += "</ul>"
    top_models_html += "</div>"

    # 5. Success Rate Evolution
    success_evolution = {model: [] for model in all_horizons_metrics[forecast_horizons[0]].keys()}
    for horizon in forecast_horizons:
        for model in success_evolution.keys():
            success_rate = (all_horizons_metrics[horizon][model]['sMAPE'] < 40).mean() * 100
            success_evolution[model].append(success_rate)
    
    fig_success = go.Figure()
    for model, rates in success_evolution.items():
        fig_success.add_trace(go.Scatter(
            x=forecast_horizons, y=rates,
            mode='lines+markers', name=model
        ))
    fig_success.update_layout(title='Success Rate Evolution (Within 40% Error)',
                            xaxis_title='Forecast Horizon (months)',
                            yaxis_title='Success Rate (%)')
    figs.append(fig_success)

    # Generate combined dashboard
    dashboard_html = "<html><head><title>Total Analysis Dashboard</title></head><body>"
    dashboard_html += top_models_html
    for fig in figs:
        dashboard_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    dashboard_html += "</body></html>"

    with open(os.path.join(total_dir, 'total_dashboard.html'), 'w') as f:
        f.write(dashboard_html)
