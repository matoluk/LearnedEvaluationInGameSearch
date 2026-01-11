import numpy as np
import math
import struct

# ============================================================
# CONFIG
# ============================================================

NAME = "gomoku4MC10000"
INPUT_CSV = NAME + ".csv"
OUTPUT_BIN = NAME + ".bin"

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE

# ============================================================
# BOARD TRANSFORMS
# ============================================================

def reshape_board(x):
    return x.reshape(BOARD_SIZE, BOARD_SIZE)

def rotations_and_flips(board):
    b = board
    for _ in range(4):
        yield b
        yield np.fliplr(b)
        b = np.rot90(b)

# ============================================================
# ENCODING
# ============================================================

def encode_board(board):
    """
    board: numpy array 15x15 with values -1,0,1
    returns: bytes(60)
    """
    out = bytearray(60)
    idx = 0

    for row in board:
        v = 0

        for cell in row:
            v *= 4
            if cell == 1:
                v += 1
            elif cell == -1:
                v += 2

        # uložíme 4 bajty (32 bitov)
        out[idx:idx+4] = v.to_bytes(4, "little")
        idx += 4

    return bytes(out)



# ============================================================
# MAIN
# ============================================================

def main():
    data = {}  # bytes(60) -> [sum_y, count]

    rows = 0
    augmented = 0

    with open(INPUT_CSV, "r") as f:
        for line in f:
            rows += 1

            row = np.fromstring(line, sep=",", dtype=np.float32)
            x = row[:-1]
            y = float(row[-1])

            board = reshape_board(x)

            for sign in (1, -1):
                b_signed = board * sign
                y_signed = y * sign

                for b in rotations_and_flips(b_signed):
                    key = encode_board(b)

                    if key in data:
                        data[key][0] += y_signed
                        data[key][1] += 1
                    else:
                        data[key] = [y_signed, 1]

                    augmented += 1
                    if (augmented & (augmented - 1)) == 0:
                        print(f"augmented: {augmented}")

    print(f"Original positions: {rows}")
    print(f"After augmentation: {augmented}")
    print(f"Unique positions: {len(data)}")

    # ========================================================
    # WRITE BIN FILE
    # ========================================================

    with open(OUTPUT_BIN, "wb") as out:
        for board_bytes, (s, c) in data.items():
            out.write(board_bytes)
            out.write(struct.pack("f", s / c))

    print(f"Saved {len(data)} records")
    print(f"File: {OUTPUT_BIN}")
    print(f"Record size: {RECORD_SIZE} bytes")

if __name__ == "__main__":
    main()
