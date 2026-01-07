import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PositionGomoku implements Position {
    final int size;
    final int[] board;
    int deep;
    GameState state;
    static class Move implements Comparable {
        int x = 0;
        int y = 0;
        Move() {}
        Move(int x, int y) {
            this.x = x;
            this.y = y;
        }
        Move(Move m1, int mul, Move m2) {
            x = m1.x + mul * m2.x;
            y = m1.y + mul * m2.y;
        }
        Move next(int size){
            if (x == size - 1 && y == size - 1)
                return null;
            return new Move(y == size - 1 ? x + 1 : x, (y + 1) % size);
        }
        void add(Move move) {
            x += move.x;
            y += move.y;
        }
        boolean isValid(int size) {
            return x >= 0 && y >= 0 && x < size && y < size;
        }
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof Move m))
                return false;
            return x == m.x && y == m.y;
        }
        @Override
        public int hashCode() {
            return Integer.hashCode(x) * 31 + Integer.hashCode(y);
        }

        @Override
        public String toString() {
            return "[" + x + "," + y + "]";
        }

        @Override
        public int compareTo(Object o) {
            if (!(o instanceof Move m))
                return -1;
            return 20 * x + y - (20 * m.x + m.y);
        }
    }
    PositionGomoku(int size) {
        this.size = size;
        board = new int[this.size];
        deep = 0;
        state = GameState.ONGOING;
    }
    PositionGomoku(PositionGomoku pos) {
        size = pos.size;
        board = Arrays.copyOf(pos.board, size);
        deep = pos.deep;
        state = pos.state;
    }

    @Override
    public Position copy() {
        return new PositionGomoku(this);
    }

    int get(Move move) {
        return ((board[move.x] >> (move.y * 2)) & 3);
    }
    static int get(int[] board, Move move) {
        return ((board[move.x] >> (move.y * 2)) & 3);
    }
    void set(Move move, int pl) {
        board[move.x] = (board[move.x] & ~(3 << (move.y * 2))) | (pl << (move.y * 2));
    }
    static void set(int[] board, Move move, int pl) {
        board[move.x] = (board[move.x] & ~(3 << (move.y * 2))) | (pl << (move.y * 2));
    }
    int actualPlayer() {
        return (deep & 1) + 1;
    }
    @Override
    public List<Object> moves() {
        List<Object> moves = new ArrayList<>();
        for (Move move = new Move(); move != null; move = move.next(size))
            if (get(move) == 0)
                moves.add(move);
        return moves;
    }

    @Override
    public GameState state() {
        return state;
    }
    private GameState move(Move move) {
        if (get(move) != 0)
            throw new RuntimeException("Square " + move + " is not free.");

        int player = actualPlayer();
        set(move, player);
        deep++;

        for (Move dir : List.of(new Move(1,0), new Move(0,1), new Move(1,1), new Move(1,-1))) {
            int count = 0;
            Move end = new Move(move, 5, dir);
            for (Move square = new Move(move, -4, dir); !square.equals(end); square.add(dir)) {
                if (square.isValid(size)) {
                    if (get(square) == player)
                        count++;
                    else
                        count = 0;
                    if (count >= 5)
                        return GameState.WIN;
                }
            }
        }
        if (deep >= size * size)
            return GameState.DRAW;
        return GameState.ONGOING;
    }
    @Override
    public Position applyMove(Object move) {
        if (!(move instanceof Move m))
            throw new IllegalArgumentException("Expected gomoku.Move");
        state = move(m);
        return this;
    }

    @Override
    public void revertMove(Object move) {
        if (!(move instanceof Move m))
            throw new IllegalArgumentException("Expected gomoku.Move");
        set(m, 0);
        state = GameState.ONGOING;
        deep--;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof PositionGomoku o) || deep != o.deep || size != o.size || state != o.state)
            return false;
        if (Arrays.equals(board, o.board))
            return true;

        int[] other = o.board;
        int[] mirror = new int[size];
        int maxIndex = size - 1;
        for (int i = 0; i < size; i++)
            mirror[i] = other[maxIndex - i];
        if (Arrays.equals(board, mirror))
            return true;
        
        for (int i = 0; i < 3; i++) {
            int[] rotateOther = new int[size];
            int[] rotateMirror = new int[size];
            for (Move move = new Move(); move != null; move = move.next(size)) {
                @SuppressWarnings("SuspiciousNameCombination")
                Move rotate = new Move(maxIndex - move.y, move.x);
                set(rotateOther, rotate, get(other, move));
                set(rotateMirror, rotate, get(mirror, move));
            }
            if (Arrays.equals(board, rotateOther) || Arrays.equals(board, rotateMirror))
                return true;
            other = rotateOther;
            mirror = rotateMirror;
        }
        return false;
    }
    public static int hash(int x) {
        x ^= (x >>> 16);
        x *= 0x85ebca6b;
        x ^= (x >>> 13);
        x *= 0xc2b2ae35;
        x ^= (x >>> 16);
        return x;
    }
    @Override
    public int hashCode() {
        int hash = 0;
        for (Move move = new Move(); move != null; move = move.next(size)) {
            if (get(move) == 0)
                continue;
            int x = move.x >= size / 2 ? size - 1 - move.x : move.x;
            int y = move.y >= size / 2 ? size - 1 - move.y : move.y;
            int code = Integer.max(x, y);
            code += size / 2 * (x + y - code) + 1;
            if (get(move) == 2)
                code += size * size / 4 + 1;
            hash += hash(code);
        }
        return hash;
    }

    @Override
    public String toString() {
        String[] figures = {". ", "X ", "O ", "Er"};
        StringBuilder stringBuilder = new StringBuilder();
        for (Move move = new Move(); move != null; move = move.next(size)) {
            stringBuilder.append(figures[get(move)]);
            if (move.y == size - 1 && move.x < size - 1)
                stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }

    @Override
    public String flattenBoard() {
        StringBuilder sb = new StringBuilder();
        int me = actualPlayer();

        for (Move m = new Move(); m != null; m = m.next(size)) {
            int v = get(m);

            if (v == 0) {
                sb.append("0,");
            } else if (v == me) {
                sb.append("1,");
            } else {
                sb.append("-1,");
            }
        }
        return sb.toString();
    }
}
