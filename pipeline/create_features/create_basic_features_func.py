import polars as pl
import pandas as pd
from configs.history_data_crawlers_config import root_path, symbols_dict
from datetime import datetime, timedelta
from utils.logging_tools import default_logger
from pathlib import Path
from feature_engine.datetime import DatetimeFeatures



def add_candle_features(
    df, symbol: str, tf_list=[5, 15, 60, 120, 240, 1440], fe_prefix="fe_cndl"
):
    org_columns = set(df.columns)
    org_columns.remove("_time")
    for tf in tf_list:
        continue

    return df

def add_shifted_candle_features(
    df, tf, shift_sizes=[1], fe_prefix="fe_cndl_shift"
):
    org_columns = set(df.columns)
    org_columns.remove("_time")
    for sh in shift_sizes:
        continue

    return df

def history_basic_features(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_basic_features fumc:")
    try:

        fe_prefix = "fe_cndl"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)


        for symbol in list(feature_config.keys()):
            continue

        # ----------------------------------
        fe_prefix = "fe_cndl_shift"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)

        for symbol in list(feature_config.keys()):
            if fe_prefix not in list(feature_config[symbol].keys()):
                continue

            logger.info(f"- " * 30)
            logger.info(f"--> {fe_prefix}, {symbol} :")
            
            df_sh = pl.read_parquet(
                f"{root_path}/data/features/fe_cndl_shift_raw/{fe_prefix}_{symbol}.parquet"
            )

            file_name = features_folder_path + f"/{fe_prefix}_{symbol}.parquet"
            shift_configs = feature_config[symbol]["fe_cndl_shift"]["shift_configs"]
            
            df_sh = df_sh.sort("_time")
            df_all = df_sh[["_time"]]

            for shift_config in shift_configs:
                timeframe = shift_config["timeframe"]

                shift_sizes = shift_config["shift_sizes"]
                df = add_shifted_candle_features(
                    df_sh,
                    tf=timeframe,
                    shift_sizes=shift_sizes,
                    fe_prefix=fe_prefix,
                )
                

            
            df_all = df_all.with_columns(pl.lit(symbol).alias("symbol"))
            df_all.write_parquet(file_name)
            logger.info(f"--> fe_cndl_shift_stage_2 {symbol} saved.")

        logger.info("--> history_basic_features run successfully.")
    except Exception as e:
        logger.exception("--> history_basic_features error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")


def history_fe_time(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start history_fe_time fumc:")
    try:
        prefix = "fe_time"
        symbol = list(feature_config.keys())[0]
        df= pd.read_parquet(f"{root_path}/data/stage_one_data/EURUSD_stage_one.parquet", columns=["_time"]).sort_values("_time").reset_index(drop=True)
        df["_time"] = df["_time"].dt.tz_localize(None)
        df.sort_values("_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["_time"] = df["_time"] - timedelta(hours=7)

        dtf = DatetimeFeatures(
            features_to_extract=[
                "month",
                "quarter",
                "semester",
                "week",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "month_start",
                "month_end",
                "quarter_start",
                "quarter_end",
                "year_start",
                "year_end",
                "hour",
                "minute",
            ],
            drop_original=False,
        )

        dtf.fit(df.rename(columns={"_time": prefix}))
        fe_time = dtf.transform(df.rename(columns={"_time": prefix})).rename(
            columns={prefix: "_time"}
        )
    
        markets_trade_times = {
            "New_York": (8, 17),
            "Tokyo": (19, 4),
            "Sydney": (15, 0),
            "London": (3, 11),
        }
        
        for market_name in markets_trade_times:
            start_time = markets_trade_times[market_name][0]
            stop_time = markets_trade_times[market_name][1]

            col_name = f"{prefix}_isin_{market_name}"
            fe_time[col_name] = 0

            if stop_time > start_time:
                fe_time.loc[
                    (fe_time["_time"] >= start_time)
                    & (fe_time["_time"] < stop_time),
                    col_name,
                ] = 1

            else:
                fe_time.loc[
                    (fe_time["_time"].dt.hour >= start_time)
                    | (fe_time["_time"].dt.hour < stop_time),
                    col_name,
                ] = 1

        fe_time["_time"] = fe_time["_time"] + timedelta(hours=7)
        fe_time["symbol"] = symbol
        features_folder_path = f"{root_path}/data/features/{prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        file_name = features_folder_path + f"/{prefix}.parquet"
        fe_time.to_parquet(file_name,index=False)

        logger.info("--> history_fe_time run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_time error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")


if __name__ == "__main__":
    from utils.config_utils import read_feature_config
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()

    history_basic_features(config_general)
    print(f"--> history_basic_features DONE.")

    history_fe_market_close(config_general)
    print(f"--> read_feature_config DONE.")

    history_fe_time(config_general)
    print(f"--> read_feature_config DONE.")
