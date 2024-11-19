import polars as pl
import numpy as np
import pandas as pd
import datetime
from configs.history_data_crawlers_config import symbols_dict
from pyarrow.parquet import ParquetFile
from target_generation import calculate_classification_target_numpy_ver
import time
from sklearn.feature_selection import VarianceThreshold
from configs.stage_one_data_config import stage_one_data_path
def ETL(
    path,# path of dataset
    stage_one_data_path,
    trade_mode,
    target_symbol,
    trg_look_ahead,
    trg_take_profit,
    trg_stop_loss,
    n_rand_features,
    target_col,# name of target column
    base_time_frame,# for calculating targerts
):
    raw_columns = [f.name for f in ParquetFile(path).schema]
    print(f'Len all columns in dataframe is {len(raw_columns)}')
    df = pd.read_parquet(path)
    print(f'Len read columns is {df.shape[1]}')
    print("Calculating target --->")
    window_size = int(trg_look_ahead // base_time_frame)
    
    df_raw = pd.read_parquet(stage_one_data_path, 
      columns = [
        '_time',
        "low",
      ]
    ).rename(columns={
        "close":f"{target_symbol}_M5_CLOSE",
        "high":f"{target_symbol}_M5_HIGH",
        "low":f"{target_symbol}_M5_LOW",
    })

    
    
    
    array = df.merge(df_raw, on = '_time', how = 'left')[
        [f"{target_symbol}_M5_CLOSE", f"{target_symbol}_M5_HIGH", f"{target_symbol}_M5_LOW"]
    ].to_numpy()
    tic = time.time()
    df["target"] = calculate_classification_target_numpy_ver(
            array,
            window_size,
            symbol_decimal_multiply = symbols_dict[target_symbol]["pip_size"],
            take_profit = trg_take_profit,
            stop_loss = trg_stop_loss,
            mode = trade_mode,
        )
    toc = time.time()
    df.dropna(inplace = True)
    print(f"---> Target {target_col} has been generated in {toc-tic:.2f} seconds")
    print("df shape: ", df.shape)
    df.set_index(["_time"], inplace=True, drop=True)
    df["target"] = df["target"].astype(int)

    ##? set targets to 0 in bad hours 
    #TODO: CHECK
    df.loc[(df.index.get_level_values('_time').time>=datetime.time(0, 0))&(df.index.get_level_values('_time').time<=datetime.time(1, 0)),'target'] = 0
    df = remove_future_redundendat_columns(df)


    print("=" * 30)
    print("--> df final shape:", df.shape)
    print(
        f"--> df min_time: {df.index.get_level_values('_time').min()} | df max_time: {df.index.get_level_values('_time').max()}"
    )
    print(
        f"--> number of unique days: {df.index.get_level_values('_time').unique().shape[0]}"
    )
    print("=" * 30)
    return df


def remove_future_redundendat_columns(df_all):
    """
    get dataframe and remove listed futures(cols) and return the dataframe
    """

    other_target_cols = [col for col in df_all.columns if "trg_" in col]

    if len(other_target_cols) > 0:
        print("columns_removed: ", other_target_cols)
    
    
    df_all = df_all.drop(columns=other_target_cols, errors="ignore")
    
    from sklearn.feature_selection import VarianceThreshold
    
    # #?? DROP constant columns:
    # print("--> DROP constant columns.")
    # sel = VarianceThreshold(threshold=0.01) # 0.1 indicates 99% of observations approximately
    # sel.fit(df_all)  # fit finds the features with zero variance
    # constant_cols = [x for x in df_all.columns if x not in df_all.columns[sel.get_support()]]
    # df_all.drop(columns=constant_cols,inplace=True)
    

    return df_all

