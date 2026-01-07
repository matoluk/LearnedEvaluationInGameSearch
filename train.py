import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load(path):
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    print(f"Dataset size: {len(raw)}")

    X = raw[:, :-1]
    y = raw[:, -1]

    return X, y

class GomokuDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ValueNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)

def main():
    NAME = "gomoku64MCAB500"
    DATA_PATH = NAME + "a.csv"
    BATCH_SIZE = 256
    EPOCHS = 30
    LR = 1e-3
    MODEL_PATH = NAME+".pt"
    MODEL_STATE_PATH = NAME + "_state.pt"

    X, y = load(DATA_PATH)

    dataset = GomokuDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ValueNet(input_size=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | loss = {loss:.5f}")

    # save PyTorch
    torch.save(model.state_dict(), MODEL_STATE_PATH)
    print("State dict saved to", MODEL_STATE_PATH)

    # save TorchScript
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(MODEL_PATH)
    print("TorchScript model saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
