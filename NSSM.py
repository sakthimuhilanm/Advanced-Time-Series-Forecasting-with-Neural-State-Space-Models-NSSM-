# Full corrected NSSM pipeline (copy & run)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 1) Data generation (same as original)
np.random.seed(42)
T = 1200
time = np.arange(T)

seasonality1 = 10 * np.sin(2 * np.pi * time / 24)
seasonality2 = 5 * np.cos(2 * np.pi * time / 12)
trend1 = 0.01 * time ** 1.5
trend2 = np.log1p(time)
feature3 = np.sin(0.1 * time) * trend1
feature4 = np.sqrt(time) + np.random.normal(0, 0.5, T)
feature5 = np.tanh(0.05 * time) + np.random.normal(0, 0.2, T)

data = pd.DataFrame({
    'var1': seasonality1 + trend1 + np.random.normal(0, 1, T),
    'var2': seasonality2 + trend2 + np.random.normal(0, 1, T),
    'var3': feature3 + np.random.normal(0, 0.5, T),
    'var4': feature4 + np.random.normal(0, 0.5, T),
    'var5': feature5 + np.random.normal(0, 0.5, T),
})

# 2) Prepare data
values = data.values
train_size = int(0.8 * T)
train_data = values[:train_size]
test_data = values[train_size:]

scaler = StandardScaler()
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

sequence_length = 30

def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 3) Models (Encoder, Transition, Observation, NSSM)
class Encoder(nn.Module):
    def __init__(self, obs_dim, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x):
        return self.net(x)

class StateTransition(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x):
        return self.net(x)

class ObservationModel(nn.Module):
    def __init__(self, state_dim, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    def forward(self, x):
        return self.net(x)

class NSSM(nn.Module):
    def __init__(self, state_dim, obs_dim, hidden_dim=64):
        super().__init__()
        self.encoder = Encoder(obs_dim, state_dim, hidden_dim)
        self.transition = StateTransition(state_dim, hidden_dim)
        self.observation = ObservationModel(state_dim, obs_dim, hidden_dim)

    def encode_initial_state(self, obs_seq):
        last_obs = obs_seq[:, -1, :]
        return self.encoder(last_obs)

    def forward(self, obs_seq, n_steps=1):
        state = self.encode_initial_state(obs_seq)
        preds = []
        for _ in range(n_steps):
            state = self.transition(state)
            y_pred = self.observation(state)
            preds.append(y_pred)
        preds = torch.stack(preds, dim=1)
        return preds

# 4) Train NSSM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dim = 8
hidden_dim = 64
obs_dim = train_scaled.shape[1]

nssm = NSSM(state_dim, obs_dim, hidden_dim).to(device)
optimizer = optim.Adam(nssm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

num_epochs = 30
for epoch in range(num_epochs):
    nssm.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = nssm(batch_x, n_steps=1)[:, 0, :]
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    epoch_loss = total_loss / len(dataset)
    if (epoch+1) % 5 == 0 or epoch==0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

# 5) Autoregressive forecast with NSSM
def forecast_nssm_autoregressive(model, seed_sequence, forecast_horizon):
    model.eval()
    seq = seed_sequence.copy()
    preds = []
    with torch.no_grad():
        for _ in range(forecast_horizon):
            seq_tensor = torch.tensor(seq[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seq_tensor, n_steps=1)[:, 0, :].cpu().numpy().squeeze()
            preds.append(pred)
            seq = np.vstack([seq, pred])
    return np.array(preds)

seed_sequence = test_scaled[:sequence_length]
forecast_horizon = 50
nssm_preds_scaled = forecast_nssm_autoregressive(nssm, seed_sequence, forecast_horizon)
nssm_preds_rescaled = scaler.inverse_transform(nssm_preds_scaled)

# 6) Benchmark: faster ARIMA per series (original scale)
models = []
for i in range(train_data.shape[1]):
    try:
        model = sm.tsa.ARIMA(train_data[:, i], order=(2,1,2))
        model_fit = model.fit()
    except Exception:
        model_fit = None
    models.append(model_fit)

arima_preds = []
for i, model_fit in enumerate(models):
    if model_fit is None:
        pred = np.repeat(train_data[-1, i], forecast_horizon)
    else:
        pred = model_fit.forecast(steps=forecast_horizon)
    arima_preds.append(pred)
arima_preds = np.array(arima_preds).T

# 7) Evaluate & Plot
true_vals = data.iloc[train_size+sequence_length:train_size+sequence_length+forecast_horizon].values

def evaluate_forecasts(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return mse, mae

print("NSSM Forecast:")
for i in range(obs_dim):
    mse, mae = evaluate_forecasts(true_vals[:,i], nssm_preds_rescaled[:,i])
    print(f"Variable {i+1}: MSE={mse:.4f}, MAE={mae:.4f}")

print("\nARIMA Benchmark Forecast:")
for i in range(obs_dim):
    mse, mae = evaluate_forecasts(true_vals[:,i], arima_preds[:,i])
    print(f"Variable {i+1}: MSE={mse:.4f}, MAE={mae:.4f}")

# Plotting combined (NSSM vs ARIMA)
plt.figure(figsize=(12,10))
for i in range(obs_dim):
    plt.subplot(obs_dim,1,i+1)
    plt.plot(range(len(data)), data.iloc[:,i], label='Actual')
    start_idx = train_size + sequence_length
    plt.plot(range(start_idx, start_idx+forecast_horizon), nssm_preds_rescaled[:,i], label='NSSM Forecast')
    plt.plot(range(start_idx, start_idx+forecast_horizon), arima_preds[:,i], label='ARIMA Forecast', linestyle='--')
    plt.legend(loc='upper left')
plt.suptitle('Forecasts: NSSM vs ARIMA (per series)')
plt.tight_layout()
plt.show()

