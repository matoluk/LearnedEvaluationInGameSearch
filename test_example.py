import torch
import torch.nn as nn

BOARD_SIZE = 15
INPUT_SIZE = BOARD_SIZE * BOARD_SIZE

class EvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def parse_board_input():
    print(f"Enter {BOARD_SIZE} rows of the board (use 'x', 'o', '-'):")
    board = []
    for i in range(BOARD_SIZE):
        while True:
            row = input().strip()
            if len(row) != BOARD_SIZE:
                print(f"Row must be {BOARD_SIZE} characters long. Try again.")
                continue
            valid = all(c in "xo-" for c in row)
            if not valid:
                print("Row contains invalid characters. Use only 'x', 'o', '-'")
                continue
            board.extend(1 if c=='x' else -1 if c=='o' else 0 for c in row)
            break
    return torch.tensor(board, dtype=torch.float32)

def main():
    NAME = "gomoku64MCAB500"
    model = EvalNet()
    model.load_state_dict(torch.load(NAME + "_state.pt", map_location="cpu"))
    model.eval()

    x = parse_board_input().unsqueeze(0)

    with torch.no_grad():
        value = model(x).item()

    print(f"\nEvaluation: {value:.4f}")

if __name__ == "__main__":
    main()
