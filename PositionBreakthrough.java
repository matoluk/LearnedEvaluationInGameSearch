import java.util.*;

public class PositionBreakthrough implements Position {
    final int size;
    @SuppressWarnings("unchecked")
    final Set<Integer>[] pieces = new HashSet[2]; //Hash or Tree ?
    int actualPlayer;
    GameState state;
    static class Move {
        int from;
        int to;
        boolean takes;
        Move(int from, int to, boolean takes) {
            this.from = from;
            this.to = to;
            this.takes = takes;
        }

        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof Move o))
                return false;
            return from == o.from && to == o.to && takes == o.takes;
        }

        @Override
        public int hashCode() {
            return from + 127 * to;
        }

        @Override
        public String toString() {
            return "["+from/8+","+from%8+"]->["+to/8+","+to%8+"]";
        }
    }
    PositionBreakthrough(int size) {
        this.size = size;
        pieces[0] = new HashSet<>();
        pieces[1] = new HashSet<>();

        int lastRow = size * (size - 1);
        int rowBefore = lastRow - size;
        for (int i = 0; i < size; i++) {
            pieces[0].add(i);
            pieces[0].add(size + i);
            pieces[1].add(lastRow + i);
            pieces[1].add(rowBefore + i);
        }

        actualPlayer = 0;
        state = GameState.ONGOING;
    }
    PositionBreakthrough(PositionBreakthrough pos) {
        size = pos.size;
        pieces[0] = new HashSet<>(pos.pieces[0]);
        pieces[1] = new HashSet<>(pos.pieces[1]);
        actualPlayer = pos.actualPlayer;
        state = pos.state;
    }
    @Override
    public Position copy() {
        return new PositionBreakthrough(this);
    }
    @Override
    public List<Object> moves() {
        List<Object> moves = new ArrayList<>();
        int dir = actualPlayer == 0 ? size : -size;
        for (int piece : pieces[actualPlayer]) {
            int forward = piece + dir;
            int left = forward - 1;
            int right = forward + 1;

            if (piece % size > 0 && !pieces[actualPlayer].contains(left))
                moves.add(new Move(piece, left, pieces[1 - actualPlayer].contains(left)));
            if (!pieces[1 - actualPlayer].contains(forward) && !pieces[actualPlayer].contains(forward))
                moves.add(new Move(piece, forward, false));
            if (piece % size < size - 1 && !pieces[actualPlayer].contains(right))
                moves.add(new Move(piece, right, pieces[1 - actualPlayer].contains(right)));
        }
        return moves;
    }
    @Override
    public GameState state() {
        return state;
    }

    @Override
    public Position applyMove(Object move) {
        if (!(move instanceof Move m))
            throw new IllegalArgumentException("Expected breakthrough.Move");

        pieces[actualPlayer].remove(m.from);
        pieces[actualPlayer].add(m.to);
        actualPlayer = 1 - actualPlayer;
        if (m.takes)
            pieces[actualPlayer].remove(m.to);
        if (m.to < size || m.to >= (size * (size - 1)) || pieces[actualPlayer].isEmpty())
            state = GameState.WIN;
        return this;
    }

    @Override
    public void revertMove(Object move) {
        if (!(move instanceof Move m))
            throw new IllegalArgumentException("Expected breakthrough.Move");

        if (m.takes)
            pieces[actualPlayer].add(m.to);
        actualPlayer = 1 - actualPlayer;
        pieces[actualPlayer].remove(m.to);
        pieces[actualPlayer].add(m.from);
        state = GameState.ONGOING;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof PositionBreakthrough o) || size != o.size || actualPlayer != o.actualPlayer)
            return false;
        return pieces[0].equals(o.pieces[0]) && pieces[1].equals(o.pieces[1]);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = size * (size - 1); i >= 0; i -= size) {
            for (int j = 0; j < size; j++) {
                int pos = i + j;
                if (pieces[0].contains(pos))
                    stringBuilder.append("A ");
                else if (pieces[1].contains(pos))
                    stringBuilder.append("V ");
                else
                    stringBuilder.append(". ");
            }
            stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }
    @Override
    public String flattenBoard() {
        StringBuilder sb = new StringBuilder();
        int me = actualPlayer;

        for (int i = size * (size - 1); i >= 0; i -= size) {
            for (int j = 0; j < size; j++) {
                int pos = i + j;

                if (pieces[me].contains(pos)) {
                    sb.append("1,");
                } else if (pieces[1 - me].contains(pos)) {
                    sb.append("-1,");
                } else {
                    sb.append("0,");
                }
            }
        }
        return sb.toString();
    }
}
