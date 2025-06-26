# scripts/predict.py

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import argparse
import os

# Model Definition (same as train_model.py)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# Predict next N years using autoregressive loop
def predict_next_n_years(input_csv, model_path, seq_len=5, n_years=5):
    # Load data
    df = pd.read_csv(input_csv)
    years = df["Year"].tolist()
    values = df["Total_CombinedSI"].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    data_tensor = torch.FloatTensor(scaled).view(-1)

    # Load model
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict forward
    current_seq = data_tensor[-seq_len:].clone().unsqueeze(0)
    future_years = []
    future_preds = []

    with torch.no_grad():
        for i in range(n_years):
            pred_scaled = model(current_seq).item()

            # Clamp scaled prediction to prevent inverse_transform from exploding
            pred_scaled = max(pred_scaled, 0.0)

            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]

            # Clamp real-world prediction to ensure no negative spawning index
            pred_real = max(pred_real, 0.0)

            next_year = years[-1] + 1 + i
            print(f"Predicted spawning index for {next_year}: {pred_real:.2f}")

            future_years.append(next_year)
            future_preds.append(pred_real)

            # Update input sequence with scaled prediction (not real-world)
            pred_tensor = torch.FloatTensor([pred_scaled])
            current_seq = torch.cat([current_seq.squeeze(0)[1:], pred_tensor]).unsqueeze(0)

    return future_years, future_preds

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/sog_spawn_index.csv", help="Path to processed CSV")
    parser.add_argument("--model", type=str, help="Path to trained model (.pth), overrides --seq_len")
    parser.add_argument("--seq_len", type=int, default=5, help="Input sequence length")
    parser.add_argument("--n_years", type=int, default=5, help="How many years to forecast")
    args = parser.parse_args()

    # Auto-generate model path if not provided
    model_path = args.model
    if model_path is None:
        model_path = f"models/lstm_sog_2022_seq{args.seq_len}.pth"

    predict_next_n_years(
        input_csv=args.input,
        model_path=model_path,
        seq_len=args.seq_len,
        n_years=args.n_years
    )