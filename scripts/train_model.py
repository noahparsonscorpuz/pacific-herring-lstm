import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os

# Modyify to change sequence length
SEQ_LEN = 6

# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dimension
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Take last time step
        return out

# Sequence Generation
def create_sequences(data_tensor, seq_len):
    xs, ys = [], []
    for i in range(len(data_tensor) - seq_len):
        x = data_tensor[i:i+seq_len]
        y = data_tensor[i+seq_len]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

# Training Logic
def train_model(input_csv, model_path, seq_len=5, epochs=300, lr=0.01):
    # Load data
    df = pd.read_csv(input_csv)
    values = df["Total_CombinedSI"].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    data_tensor = torch.FloatTensor(scaled).view(-1)

    # Create sequences
    X, y = create_sequences(data_tensor, seq_len)

    # Define model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop
    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    input_csv = "../data/processed/sog_spawn_index.csv"
    model_path = f"../models/lstm_sog_2022_seq{SEQ_LEN}.pth"
    train_model(input_csv, model_path, seq_len=SEQ_LEN)