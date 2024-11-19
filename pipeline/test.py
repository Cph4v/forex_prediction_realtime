import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime
import time
import sys
import os
import numpy as np

# Add path to modeling_funcs
current_dir = os.path.dirname(os.path.abspath(__file__))
modeling_path = os.path.abspath(os.path.join(current_dir, "..", "modeling"))
sys.path.append(modeling_path)

# Load parquet dataset paths
parquet_file = os.path.join(current_dir, '..', 'pipeline', 'data', 'dataset', 'dataset.parquet')
dukascopy_file = os.path.join(current_dir, 'data', 'raw_data', 'dukascopy', 'EURUSD_dukascopy.parquet')
stage_one_file = os.path.join(current_dir, 'data', 'stage_one_data', 'EURUSD_stage_one.parquet')

# Function to update dataset
def update_dataset():
    from configs.feature_configs_general import generate_general_config
    from data_crawlers.dukascopy_func import crawl_OHLCV_data_dukascopy
    from data_crawlers.metatrader_func import crawl_OHLCV_data_metatrader
    from stage_one_data.history_data_stage_one_func import history_data_stage_one
    from realtime_candle.realtime_candle_func import historiy_realtime_candle
    from create_features.indicator_func import history_indicator_calculator
    from create_features.realtime_shift_func import history_cndl_shift
    from create_features.create_basic_features_func import history_basic_features, history_fe_market_close, history_fe_time
    from create_features.window_agg_features_func import history_fe_WIN_features
    from create_dataset.columns_merge_func import history_columns_merge
    from modeling_funcs import ETL

    config_general = generate_general_config()
    crawl_OHLCV_data_dukascopy(feature_config=config_general)
    crawl_OHLCV_data_metatrader(config_general)
    history_data_stage_one(config_general)
    historiy_realtime_candle(config_general)
    history_indicator_calculator(config_general)
    history_cndl_shift(config_general)
    history_basic_features(config_general)
    history_fe_market_close(config_general)
    history_fe_time(config_general)
    history_fe_WIN_features(config_general)
    history_columns_merge(config_general, general_mode=True)

    # Update parquet dataset
    dataset_path = os.path.join(current_dir, '..', 'pipeline', 'data', 'dataset', 'dataset.parquet')
    stage_one_data_path = os.path.join(current_dir, '..', 'pipeline', 'data', 'stage_one_data', 'EURUSD_stage_one.parquet')
    df_all = ETL(
        path=dataset_path,
        stage_one_data_path=stage_one_data_path,
        trade_mode="long",
        target_symbol="EURUSD",
        trg_look_ahead=300,
        trg_take_profit=40,
        trg_stop_loss=15,
        n_rand_features=3,
        target_col="target",
        base_time_frame=5,
    )
    df_all.to_parquet(parquet_file)

def cumulative_return(stage_one_data, df_all):
    

    stage_one = stage_one_data.copy()
    target = df_all.copy()

    target = target.reset_index()
    target = target.rename(columns={'index': '_time'})

    target = target[['_time','target']]

    stage_one = stage_one[['_time','close']]

    merged_df = pd.merge(target, stage_one, on='_time', how='inner')

    merged_df['Tomorrows Returns percent'] = 0.
    merged_df['Tomorrows Returns percent'] = (( merged_df['close'] - merged_df['close'].shift(1))/merged_df['close'].shift(1)) * 100
    merged_df['Tomorrows Returns percent'] = merged_df['Tomorrows Returns percent'].shift(-1)
    merged_df['Strategy Returns'] = 0.
    merged_df['Strategy Returns'] = np.where(merged_df['target'] == 1, merged_df['Tomorrows Returns percent'], 0)
    merged_df.loc[:, 'Cumulative Market Returns percent'] = np.cumsum(merged_df['Tomorrows Returns percent'])
    merged_df.loc[:, 'Cumulative Strategy Returns'] = np.cumsum(merged_df['Strategy Returns'])
    merged_df = merged_df.dropna()
    strtg_return = merged_df['Cumulative Strategy Returns'].iloc[-1]
    market_return = merged_df['Cumulative Market Returns percent'].iloc[-1]
    return strtg_return, market_return

# Function to update and plot chart
def update_chart():
    update_dataset()
    try:
        feature_df = pd.read_parquet(parquet_file)
        ohlcv_df = pd.read_parquet(dukascopy_file)
        stage_one_df = pd.read_parquet(stage_one_file)
        strtg_return, market_return = cumulative_return(stage_one_df, feature_df)
        print(f"ohlcv_df : {ohlcv_df}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Convert UNIX timestamp in milliseconds to datetime with UTC timezone
    stage_one_df['_time'] = pd.to_datetime(stage_one_df['_time'], unit='ms', utc=True)
    
    # Align OHLCV data with feature data based on index
    merged_df = stage_one_df.copy()

    # Get the last row target from feature dataset
    last_target = feature_df['target'].iloc[-1]

    # Candlestick chart
    candlestick = go.Candlestick(
        x=merged_df['_time'],
        open=merged_df['open'],
        high=merged_df['high'],
        low=merged_df['low'],
        close=merged_df['close'],
        name='Candlestick'
    )

    # Highlight timestamps where target was 1 in the past
    highlight_indices = feature_df[feature_df['target'] == 1].index.tz_localize('UTC')
    highlight_timestamps = merged_df['_time'][merged_df['_time'].isin(highlight_indices)]
    print(f"highlight_timestamps : {highlight_timestamps}")
    print(f"stage_one_df : {stage_one_df}")
    print(f"feature_df : {feature_df}")
    
    highlight_points = go.Scatter(
        x=highlight_timestamps,
        y=merged_df['close'][merged_df['_time'].isin(highlight_indices)],
        mode='markers',
        name='Buy Signal (Past)',
        marker=dict(symbol='diamond', size=5, color='red')
    )
    # Add last prediction as a legend in the chart
    prediction_annotation = go.Scatter(
        x=[merged_df['_time'].iloc[-1]],
        y=[merged_df['close'].iloc[-1]],
        mode='markers+text',
        text=[f"Prediction: {'Buy' if last_target == 1 else 'Not Buy'}"],
        textposition='top center',
        name='Last Prediction',
        marker=dict(color='blue', size=10)
    )

    # layout = go.Layout(
    #     title='Real-Time Stock Market Dashboard',
    #     xaxis=dict(title='Timestamp'),
    #     yaxis=dict(title='Price'),
    #     xaxis_rangeslider_visible=False
    # )

    layout = go.Layout(
        title=f'Real-Time Stock Market Dashboard | Strategy Return: {strtg_return}% | Market Return: {market_return}%',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Price'),
        xaxis_rangeslider_visible=False
    )

    fig = go.Figure(data=[candlestick, prediction_annotation, highlight_points], layout=layout)
    pio.show(fig)

# Main loop for updating the chart every 5 minutes
if __name__ == '__main__':
    while True:
        update_chart()
        time.sleep(5 * 60)  # Wait for 5 minutes
