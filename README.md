<h2>Forex Price Prediction and Signal Generation System</h2>

<h3>Description</h3>
<p>This project presents a comprehensive solution for predicting forex market movements and generating actionable buy or sell signals for the next candle before its formation. Designed for both precision and adaptability, this system leverages cutting-edge data scraping, preprocessing, feature engineering, and machine learning to align with specific trading strategies.</p>

<p><strong>Key highlights of the project include:</strong></p>
<ul>
  <li><strong>Data Scraping</strong>: Real-time market data is sourced from <strong>Dukascopy</strong> and <strong>MetaTrader</strong>, ensuring up-to-date, high-quality input.</li>
  <li><strong>Feature Engineering</strong>: Advanced features are added to the raw data, including historical targets, volume, and enriched attributes essential for robust decision-making.</li>
  <li><strong>Data Preprocessing</strong>: Sophisticated preprocessing pipelines handle:
    <ul>
      <li><strong>Time Zone Normalization</strong>: Aligns data across multiple global markets.</li>
      <li><strong>Anomaly Management</strong>: Detects and removes null values and outliers to enhance data integrity.</li>
      <li><strong>Custom Target Engineering</strong>: Generates buy and sell targets based on historical data and pre-defined trading strategies.</li>
    </ul>
  </li>
  <li><strong>Model Training</strong>: A bespoke predictive model is developed, incorporating a specialized trading strategy that includes key parameters such as:
    <ul>
      <li>Take profit and stop loss levels.</li>
      <li>Adaptive trade volumes.</li>
      <li>Strategic signal generation.</li>
    </ul>
  </li>
  <li><strong>Prediction and Visualization</strong>: After training, the model predicts the buy or sell signal for the last candle in real time. The results are visually represented through <strong>matplotlib</strong> plots, providing an intuitive view of market behavior and model predictions.</li>
</ul>

<h3>Unique Selling Points</h3>
<ul>
  <li><strong>Custom Strategy Integration</strong>: Unlike generic forex prediction systems, this project integrates a tailored trading strategy into the machine learning pipeline.</li>
  <li><strong>Real-Time Execution</strong>: Combines live data scraping and processing for actionable insights.</li>
  <li><strong>Visualization for Decision Support</strong>: Outputs clear, interpretable plots, enabling traders to make data-driven decisions with confidence.</li>
</ul>

<p>This pipeline is a robust demonstration of how artificial intelligence can enhance trading strategies, streamline analysis, and provide a competitive edge in the volatile forex market.</p>

<details>
<summary><strong>modeling</strong></summary>

- **configs**
  - `feature_configs_general.py`
  - `history_data_crawlers_config.py`
  - `stage_one_data_config.py`
- **data**
  - **raw_data**
  - **stage_one_data**
- `modeling_funcs.py`
- `modeling.ipynb`
- `quant_cross_validation.py`
- `target_generation.py`
- **utils**
  - `general_utils.py`

</details>

<details>
<summary><strong>pipeline</strong></summary>

- **configs**
  - `feature_configs_general.py`
  - `history_data_crawlers_config.py`
  - `stage_one_data_config.py`
- **create_dataset**
  - `columns_merge_func.py`
- **create_features**
  - `create_basic_features_func.py`
  - `indicator_func.py`
  - `realtime_shift_func.py`
  - `window_agg_features_func.py`
- **data**
  - **raw_data**
    - `dukascopy`
- **data_crawlers**
  - `dukascopy_func.py`
  - `metatrader_func.py`
- `feature_creation_pipeline.ipynb`
- **logs**
  - `dukascopy_data_crawl.log`
  - `General_jobs.log`
- **realtime_candle**
  - `realtime_candle_func.py`
- `requirements.txt`
- **stage_one_data**
  - `history_data_stage_one_func.py`
- `test.py`
- **utils**
  - `clean_data.py`
  - `config_utils.py`
  - `datetime_utils.py`
  - `df_utils.py`
  - **feature_config_extractor**
    - `extract_config_from_features.py`
  - `logging_tools.py`
  - `reduce_memory.py`

</details>

![Forex Prediction Overview](/path/to/forex_prediction_1.png)
*Real-Time Market Dashboard - Broader View*

![Forex Prediction Zoomed](/path/to/forex_prediction_2.png)
*Real-Time Market Dashboard - Detailed View*
