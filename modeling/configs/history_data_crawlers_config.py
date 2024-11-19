from datetime import datetime
from pathlib import Path
import os

start_date_str = "2023/01/01"
# stop_date_str = "2024/07/01"
start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
# stop_date = datetime.strptime(stop_date_str, "%Y/%m/%d")
stop_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)



metatrader_number_of_days = 340
root_path = str(os.path.dirname(os.path.abspath(__file__))).replace("configs", "")
data_folder = f"{root_path}/data/raw_data/"
Path(data_folder).mkdir(parents=True, exist_ok=True)

symbols_dict = {
    # ? Majers
    "EURUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "EURUSD",
        "dukascopy_id": "EURUSD",
    },
    "AUDUSD": {
        "decimal_divide": 1e5,
        "pip_size": 0.0001,
        "metatrader_id": "AUDUSD",
        "dukascopy_id": "AUDUSD",
    }
}
