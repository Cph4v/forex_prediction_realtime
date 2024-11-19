

# symbols = [
#   "EURUSD", "USDCAD", "USDJPY", "EURJPY", "GBPUSD", "XAUUSD",
#   "AUDUSD", "NZDUSD", "USDCHF", "CADJPY", "EURGBP",
# ]


symbols = [
  "EURUSD", 
]

general_config = {
  'base_candle_timeframe': [...],
  
  'fe_ATR': {'timeframe': [...],
  'window_size': [...],
  'base_columns': ['HIGH', 'LOW']},


  'fe_RSTD': {'timeframe': [],
  'window_size': [],
  'base_columns': ['CLOSE']},


  'fe_WIN': {'timeframe': [5],
  'window_size': [],
  'base_columns': ['CLOSE']},


  'fe_cndl': [...],


  'fe_EMA': {'timeframe': [5],
  'window_size': [...],
  'base_columns': ['CLOSE']},

  'fe_SMA': {'base_columns': ['CLOSE'],
  'timeframe': [5],
  'window_size': [...]},

  'fe_RSI': {'timeframe': [...],
  'window_size': [...],
  'base_columns': ['CLOSE']},


  'fe_cndl_shift': {'columns': ['HIGH', 'CLOSE'],
  'shift_configs': [
    {'timeframe': 5, 'shift_sizes': [1]},
    {'timeframe': 15, 'shift_sizes': [1]},
    {'timeframe': 15, 'shift_sizes': [1]},
    {'timeframe': 15, 'shift_sizes': [1]},
    {'timeframe': 15, 'shift_sizes': [1]},
    {'timeframe': 15, 'shift_sizes': [1]}]},


  'fe_ratio': {'ATR': {'timeframe': [15],
    'window_size': [(15), (15)]},

  'EMA': {'timeframe': [5], 'window_size': [
    (15),
    (15),
    (15),
    ]},

  'RSI': {'timeframe': [15],
    'window_size': [(15), (15)]},

  'RSTD': {'timeframe': [240],
    'window_size': [(7, 14), (7, 30)]},

  'SMA': {'timeframe': [5], 'window_size': [
    (5 * 48, 15 * 48),
    (10 * 48, 15 * 48),
    ]}},

}

def generate_general_config(symbols=symbols,general_config=general_config):
  config_dict = {}
  for sym in symbols:
    config_dict[sym] = general_config
  return config_dict