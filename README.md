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

### Forex Price Prediction for USDJPY

#### Overview of Predictions
The first image provides a comprehensive view of the USDJPY forex pair, highlighting the predicted buy signals (marked in red) over several months. This visualization underscores the model's capability to identify potential buy opportunities in various market conditions, as seen by the dispersed red points throughout the graph.

![Forex Prediction Overview](/forex_prediction_1.png)
*Figure 1: Real-Time Market Dashboard - Broader View*

#### Detailed View of Recent Predictions
The second image zooms in on the last few weeks, offering a detailed perspective on recent predictions. The red points continue to indicate buy signals, providing insights into the model’s responsiveness to recent market shifts and its precision in pinpointing buy opportunities in a shorter timeframe.

![Forex Prediction Zoomed](/forex_prediction_2.png)
*Figure 2: Real-Time Market Dashboard - Detailed View*

These images showcase the effectiveness of the predictive model in real-time forex trading, demonstrating its utility in both broad and detailed analyses. The ability to visualize these predictions helps in verifying the model's alignment with trading strategies and market dynamics.

### Usage Instructions

To effectively utilize this forex prediction system, follow the steps outlined below:

#### Initial Setup
1. **Configure API Access**:
   - Locate the `.env` file in the root directory of the project.
   - You will need to set your own `API_TOKEN` and `API_KEY` in the `.env` file. These are essential for accessing the data sources and services that the application depends on.

2. **Install Dependencies**:
   - Ensure that Python and pip are installed on your system.
   - Open a terminal and navigate to the project directory.
   - Run the following command to install all required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

#### Running the Application
1. **Execute the Main Script**:
   - Once the setup is complete, you can start the application by running the `test.py` script. This script initializes the entire prediction system, processing data and generating predictions.
   - Execute the script by running:
     ```bash
     python test.py
     ```

#### Additional Information
- Ensure that all configurations in the `.env` and any other configuration files are correct and tailored to your specific requirements.
- The system assumes a network connection is available for accessing remote APIs and data sources as defined in your configurations.

By following these instructions, you should be able to successfully operate the forex prediction system and begin making informed trading decisions based on the model's predictions.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.


