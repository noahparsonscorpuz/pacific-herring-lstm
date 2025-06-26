import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

'''
Used to evaluate the ideal sequence length to train model with - defined by the smallest MSE.
-> 13 was found to be the ideal length when trained on 1951-2021.
-> 6 was found to be the ideal length when trained on 1951-2000.
'''

# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# Sequence generator
def create_sequences(data_tensor, seq_len):
    xs, ys = [], []
    for i in range(len(data_tensor) - seq_len):
        x = data_tensor[i:i + seq_len]
        y = data_tensor[i + seq_len]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Evaluation function
def evaluate_sequence_lengths(input_csv, max_seq_len=15, forecast_years=5):
    df = pd.read_csv(input_csv)
    values = df["Total_CombinedSI"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    data_tensor = torch.FloatTensor(scaled).view(-1)

    results = []
    for seq_len in range(3, max_seq_len + 1):
        if len(data_tensor) <= seq_len + forecast_years:
            continue

        X, y = create_sequences(data_tensor[:-forecast_years], seq_len)

        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(200):
            model.train()
            output = model(X)
            loss = criterion(output.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Predict last 5 values
        model.eval()
        preds = []
        current_seq = data_tensor[-forecast_years - seq_len:-forecast_years].clone().unsqueeze(0)
        with torch.no_grad():
            for _ in range(forecast_years):
                pred_scaled = model(current_seq).item()
                preds.append(pred_scaled)
                pred_tensor = torch.FloatTensor([pred_scaled])
                current_seq = torch.cat([current_seq.squeeze(0)[1:], pred_tensor]).unsqueeze(0)

        true_vals = data_tensor[-forecast_years:].numpy().flatten()
        pred_vals = np.array(preds)
        mse = np.mean((pred_vals - true_vals) ** 2)
        results.append((seq_len, mse))

    results_df = pd.DataFrame(results, columns=["Sequence Length", "MSE"])
    print(results_df.sort_values("MSE"))
    return results_df

if __name__ == "__main__":
    evaluate_sequence_lengths("data/processed/sog_spawn_index_1951_2000.csv", max_seq_len=20, forecast_years=23)