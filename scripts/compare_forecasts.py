import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def forecast_n_years(model_path, data_path, seq_len, start_year, n_years):
    df = pd.read_csv(data_path)
    df["Year"] = df["Year"].astype(int)
    values = df["Total_CombinedSI"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    start_idx = df[df["Year"] == start_year].index[0] - seq_len + 1
    current_seq = torch.FloatTensor(scaled[start_idx:start_idx + seq_len]).view(1, -1)

    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds = []
    with torch.no_grad():
        for _ in range(n_years):
            pred_scaled = model(current_seq).item()
            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            preds.append(pred_real)

            pred_tensor = torch.FloatTensor([pred_scaled])
            current_seq = torch.cat([current_seq.squeeze(0)[1:], pred_tensor]).unsqueeze(0)

    forecast_years = [start_year + 1 + i for i in range(n_years)]
    return forecast_years, preds

def compare_and_plot():
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    data_path = "data/processed/sog_spawn_index.csv"
    df = pd.read_csv(data_path)
    df["Year"] = df["Year"].astype(int)

    # Forecast using updated model and sequence length
    years_forecast, preds_forecast = forecast_n_years(
        model_path="models/lstm_sog_2022_seq6.pth",
        data_path=data_path,
        seq_len=6,
        start_year=2000,
        n_years=23
    )

    # Split data
    train_df = df[df["Year"] <= 2000]
    test_df = df[(df["Year"] > 2000) & (df["Year"] <= 2023)]

    # Plot
    plt.figure(figsize=(10, 6))

    # Training data (used for model)
    plt.plot(train_df["Year"], train_df["Total_CombinedSI"], label="Training Data (≤ 2000)", color="navy", linewidth=2)

    # Observed test data (not used in training)
    plt.plot(test_df["Year"], test_df["Total_CombinedSI"], label="Observed Data (2001–2023)", color="blue", linewidth=2, marker=".")

    # Forecasted data
    plt.plot(years_forecast, preds_forecast, label="Forecast (2001–2023)", color="red", linewidth=2, marker=".")

    # Labels and styling
    plt.title("Pacific Herring Spawning Index (SoG): Forecast vs Observed (2001–2023)")
    plt.xlabel("Year")
    plt.ylabel("Spawning Index")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    from sklearn.metrics import mean_squared_error, r2_score

    # Compute metrics
    mse = mean_squared_error(test_df["Total_CombinedSI"].values, preds_forecast)
    r2 = r2_score(test_df["Total_CombinedSI"].values, preds_forecast)

    # R2 and MSE Annotation
    metrics_text = f"$R^2$: {r2:.2f}   MSE: {mse:.3f}"
    plt.text(
        0.98, 0.02, f"R²: {r2:.2f}   MSE: {mse:.0f}",
        fontsize=10,
        transform=plt.gca().transAxes,  # place relative to axes
        ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='black')
    )
    plt.show()

if __name__ == "__main__":
    compare_and_plot()