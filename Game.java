import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.List;

public class Game {
    private static final long TIME_GRANULARITY = 15_625_000;
    private static final long TIME_PER_TURN = 128 * TIME_GRANULARITY; //64 -> 1sec
    public static final ThreadMXBean bean = ManagementFactory.getThreadMXBean();
    static void endGame(String msg) {
        System.out.println(msg);
        System.exit(0);
    }
    public static void main(String[] args) {
        List<Engine> engines = List.of(
                new EngineAB(new HeuristicGomokuNN("gomoku64MCAB500.pt"), new ABGomokuSortMoves()),
                new EngineAB(new HeuristicGomokuNN("gomoku64MCAB500.pt"), new ABGomokuSortMoves())
        );
        Position position = new PositionGomoku(15);
        int plOnTurn = 0;

        while (position.state() == GameState.ONGOING) {
            long startTime = bean.getCurrentThreadCpuTime();
            Object move = engines.get(plOnTurn).choseMove(position.copy(), startTime + TIME_PER_TURN);
            long duration = bean.getCurrentThreadCpuTime() - startTime;

            System.out.println("Turn duration: " + duration / 1_000_000.0 + "ms");
            if (duration > TIME_PER_TURN)
                endGame("Player " + (1 - plOnTurn) + " wins. Time out.");
            if (!position.moves().contains(move))
                endGame("Player " + (1 - plOnTurn) + " wins. Illegal move.");

            position.applyMove(move);
            plOnTurn = 1 - plOnTurn;
            System.out.println(position);
        }
        if (position.state() == GameState.WIN)
            endGame("Player " + (1 - plOnTurn) + " wins.");
        else
            endGame("Draw.");
    }
}
