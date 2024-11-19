import polars as pl
import datetime
from configs.stage_one_data_config import stage_one_data_path
from utils.logging_tools import default_logger
from configs.history_data_crawlers_config import root_path
from pathlib import Path

def read_and_prepeare_dataframe_polars(symbol):
    file_name = f"{stage_one_data_path}/{symbol}_stage_one.parquet"
    df = pl.read_parquet(
        file_name, columns=["_time", "open", "close", "low", "high", "tick_volume"]
    )
    df = df.sort("_time")

    df = df.rename(
        {
            "open": f"OPEN",
            "close": f"CLOSE",
            "low": f"LOW",
            "high": f"HIGH",
            "tick_volume": f"VOLUME",
        }
    )

    first_row_time = df.row(0, named=True)["_time"]
    if datetime.time(0, 0) < first_row_time.time():
        df = df.filter(
             ...
        )  # Delete the first day that started from the middle of the day

    df = df.with_columns((pl.col("_time").dt.date()).alias("_date"))
    df = df.with_columns(
        (
            ...
        ).alias("minutesPassed")
    )

    df = df.with_columns(
        pl.when(pl.col("_time").dt.time().is_in(pl.time(0, 0, 0)))
        .alias("isFirst")
    )

    df = df.with_row_index().with_columns(
        pl.col("index").cast(pl.Int32, strict=False).alias("index")
    )  # add index col
    return df
def make_realtime_candle(df, tf_list, symbol):
    # opt
    for tf_int in tf_list:
        tf_str = str(tf_int)
        df = (
            df.set_sorted("_time")
            .rolling(index_column="_time", period=tf_str + "m")
            .agg(
                [
                    pl.all().exclude("_time").last(),
                    pl.col(f"M5_OPEN")
                    .slice(
                        ...
                    )
                    .first()
                    .alias("M" + tf_str + "_OPEN")
                ]
            )
        )
    return df

def historiy_realtime_candle(feature_config, logger=default_logger):
    logger.info("- " * 25)
    logger.info("--> start historiy_realtime_candle fumc:")
    try:
        folder_path = f"{root_path}/data/realtime_candle/"
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        for symbol in list(feature_config.keys()):
            logger.info("= " * 35)
            logger.info(f"symbol: {symbol}")

            file_name = folder_path + f"{symbol}_realtime_candle.parquet"
            df = read_and_prepeare_dataframe_polars(symbol=symbol)
            tf_list = feature_config[symbol]["base_candle_timeframe"]
            df = make_realtime_candle(df, tf_list=tf_list, symbol=symbol)
            assert (
                df.filter((pl.col("isFirst") == 1) & (pl.col("minutesPassed") != 0)).shape[
                    0
                ]
                == 0
            ), "!!! bug in algorithm, all days must start at 00:00:00 time."
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            df.write_parquet(file_name)
            logger.info(f"--> {symbol} saved.")

  
        logger.info("--> historiy_realtime_candle run successfully.")
    except Exception as e:
        logger.exception("--> historiy_realtime_candle error.")     
        logger.exception(f"--> error: {e}")     
        raise ValueError("!!!")

if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    historiy_realtime_candle(config_general)
    default_logger.info(f"--> historiy_realtime_candle DONE.")
