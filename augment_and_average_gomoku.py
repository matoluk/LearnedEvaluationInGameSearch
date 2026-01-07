import numpy as np
import math
from collections import defaultdict

def reshape_board(x):
    size = int(math.sqrt(len(x)))
    return x.reshape(size, size)

def flatten_board(b):
    return b.reshape(-1)

def rotations_and_flips(board):
    boards = []
    b = board
    for _ in range(4):
        boards.append(b)
        boards.append(np.fliplr(b))
        b = np.rot90(b)
    return boards

def main():
    NAME = "gomoku64MCAB500"

    # load
    raw = np.loadtxt(NAME + ".csv", delimiter=",")
    print(f"Original rows: {len(raw)}")

    X = raw[:, :-1]
    y = raw[:, -1]

    # rotations
    data = defaultdict(list)

    for xi, yi in zip(X, y):
        board = reshape_board(xi)

        for sign in (1, -1):
            b_signed = board * sign
            y_signed = yi * sign

            for b in rotations_and_flips(b_signed):

                key = tuple(flatten_board(b))
                data[key].append(y_signed)

    print(f"After augmentation: {sum(len(v) for v in data.values())}")
    print(f"Unique positions: {len(data)}")

    # avg
    X_out = []
    y_out = []

    for board_flat, values in data.items():
        X_out.append(board_flat)
        y_out.append(np.mean(values))

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.float32)

    # save
    out = np.hstack([X_out, y_out[:, None]])
    np.savetxt(NAME + "a.csv", out, delimiter=",")

    print("Saved to " + NAME + "a.csv")


if __name__ == "__main__":
    main()
