# Pacific Herring Spawning Forecast (Strait of Georgia)

This project uses a Long Short-Term Memory (LSTM) neural network to forecast Pacific Herring spawning biomass in British Columbia’s Strait of Georgia (SoG), based on historical data from 1951–2023.

---

### Why Focus on the Strait of Georgia (SoG) and Pacific Herring?

The Strait of Georgia (SoG) herring stock is one of the most closely monitored and ecologically significant forage fish populations in Canada. It serves as an ideal case study for the following reasons:

1. **Rich Data History**: The SoG is one of the few herring regions with a consistent long-term spawning biomass dataset dating back to the 1950s.
2. **High Ecological and Economic Importance**: Pacific Herring are a keystone species, supporting salmon, seabirds, and marine mammals. They are also vital to both commercial fisheries and Indigenous cultural practices.
3. **Policy Relevance**: The SoG herring fishery is at the center of ongoing debates regarding sustainability, quotas, and conservation policy, making it a region of actionable interest.
4. **Strategic Scope and Learning Value**: This project focuses narrowly and intentionally on one region and species: Pacific Herring in the Strait of Georgia. Rather than pursuing a broad, generalized approach across multiple areas or taxa. This tight scope helped ensure that the forecasting pipeline was manageable, interpretable, and grounded in real ecological context. Furthermore, I plan to extend the processing pipeline to:
- Other forage or commercial species (e.g., salmon, halibut)
- Broader comparative studies across coastal B.C.
- ... and more!

--

## 🐟 Data Source

The dataset originates from Fisheries and Oceans Canada (DFO) and was accessed via the Pacific Salmon Foundation’s open data portal:

**[Pacific Herring Spawn Index Data (1951–2021) – Strait of Georgia](https://hub.arcgis.com/datasets/psfmarinedata::pacific-herring-spawn-index-data-altered-1951-2021-sog/about)**  
Published by: *Pacific Salmon Foundation* (via ArcGIS Hub)  
Data Owner: *Fisheries and Oceans Canada (DFO)*  
License: *Open Government License – Canada*

The data includes annual spawning index observations (`CombinedSI`) for the Strait of Georgia (SoG) Pacific Herring stock from 1951 to 2021.
Values for 2022 and 2023 were manually appended from public DFO bulletins and reports.

These were pre-processed to compute total annual spawning index values (`Total_CombinedSI`), which were saved as:

📄 data/processed/sog_spawn_index.csv

🔍 Note: This dataset represents a rare, long-term ecological time series — making it ideal for time-dependent modeling such as LSTMs.

---

## 🔧 Preprocessing

- The spawning index values were **normalized using MinMaxScaler** from scikit-learn.
- Data was split into:
  - **Training set**: 1951–2000  
  - **Test/Validation/Forecasting window**: 2001–2023
- Sequences of historical data were created for LSTM input. Each sequence is a rolling window of years feeding into the model to predict the next.

---

## 🔍 Finding Optimal Sequence Length

To determine the best number of previous time steps (`sequence length`) to use for prediction, models were trained with various sequence lengths from 3 to 20.  
The Mean Squared Error (MSE) was calculated for each configuration.

🧪 **Result**:
```text
Best performing sequence length: 6
Lowest MSE: 0.050928
```

---

## 🧠 Model Training

A simple single-layer LSTM was trained on the 1951–2000 data using the optimal sequence length of **6**.

📊 Training progress:
```
Epoch 0    - Loss: 0.2066
Epoch 50   - Loss: 0.0286
Epoch 100  - Loss: 0.0220
Epoch 150  - Loss: 0.0182
Epoch 200  - Loss: 0.0155
Epoch 250  - Loss: 0.0101
```

Final model saved to:  
📁 `models/lstm_sog_2022_seq6.pth`

---

## 📈 Forecast & Evaluation

We forecasted the **entire post-training window: 2001–2023**, then compared the predictions with actual spawning data.

### Model Performance:
- **R² Score**: 0.34  
- **Mean Squared Error (MSE)**: 64,056,801.86

👉 While modest, this R² is quite reasonable given the stochasticity and external pressures (e.g., overfishing, ecological collapse) affecting fish stocks.

---

## 📊 Visualization

The final plot compares:
- **Training Data** (≤ 2000) — blue
- **Observed Spawning Index (2001–2023)** — light blue
- **Model Forecast (2001–2023)** — red

![Forecast vs Actual](visualizations/Forecast%20vs%20Observed%20(2001-2023).png)

*Model trained on 1951–2000 data only.*

---

## ✍️ Notable Years

- **2007**: Model misses a local minimum — a rare deviation that stands out against surrounding trends.
- **2014&2017**: Model underpredicts two sharp peaks, suggesting limitations in forecasting extreme events.

---

## 🗂️ Project Structure

```
├── data/
│   └── processed/sog_spawn_index.csv
├── models/
│   └── lstm_sog_2022_seq6.pth
├── scripts/
│   ├── train_model.py
│   ├── compare_forecasts.py
│   └── tune_sequence_length.py
└── README.md
```

---

## 📌 Requirements

All dependencies are listed in [requirements.txt](requirements.txt).

Install with:
```bash
pip install -r requirements.txt
```

---

## 👋 Author's Notes

This project not only deepened my understanding of ecological forecasting, but also helped solidify my grasp of neural networks and time series modeling.

Noah Parsons Corpuz  
Computer Science & Health Information Science BSc — University of Victoria

Project: *Forecasting Pacific Fish Stocks with Deep Learning*