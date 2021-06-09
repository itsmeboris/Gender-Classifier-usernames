import pandas as pd
import plotly.graph_objects as go

for model_name in ['default', 'LSTM_CONV', 'gru_best', 'bid_GRU_bid_LSTM', 'bidirectional_LSTM']:
    results = pd.read_csv(f'results_{model_name}.csv', index_col=0)
    results = results.transpose()

    fig = go.Figure()
    for classifier in results.columns:
        fig.add_trace(go.Bar(
            x=results.index,
            y=results[classifier],
            name=f'{classifier} classifier'
        ))
    fig.update_layout(
        barmode='group',
        title="Classifiers accuracy on different data sets",
        xaxis_title="Data set",
        yaxis_title="Accuracy",
    )
    fig.write_html(f'stacked_bar_plot_{model_name}.html')
    fig.show()