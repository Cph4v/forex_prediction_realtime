import numpy as np


def calculate_classification_target_numpy_ver(
    array,
    window_size,
    symbol_decimal_multiply: float = 0.0001,
    take_profit: int = 70,
    stop_loss: int = 30,
    mode: str = "long",
):
    target_list = []

    if mode == "long":
        for i in range(array.shape[0] - window_size):
            selected_chunk = array[i : i + window_size]

            pip_diff_close = (
                selected_chunk[5:, 0] - selected_chunk[5, 0]
            ) / symbol_decimal_multiply
            pip_diff_low = (
                selected_chunk[5:, 10] - selected_chunk[5, 0]
            ) / symbol_decimal_multiply

            # BUY CLASS
            target = 0

            buy_tp_cond = pip_diff_close >= take_profit
            buy_sl_cond = pip_diff_low <= -stop_loss
            continue

        target_list.append(target)

    elif mode == "short":
        for i in range(array.shape[0] - window_size):
            continue

    for _ in range(window_size):
        target_list.append(None)

    return target_list


