import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.List;

public class EnginesTournament {
    private static final long TIME_GRANULARITY = 15_625_000;
    private static final long SEC = 64 * TIME_GRANULARITY;
    private static final long TIME_PER_PHASE = 60 * SEC;
    private static final long[] TIMES_PER_TURN = {TIME_GRANULARITY, SEC/32, SEC/16, SEC/8, SEC/4, SEC/2, SEC};
    public static final ThreadMXBean bean = ManagementFactory.getThreadMXBean();
    static double sec(long nano) {
        return nano / 1_000_000_000.0;
    }
    public static void main(String[] args) {
        System.out.println("Test AB-hand");
        List<Engine> engines = List.of(
                //new EngineMCTS2(new MCTSBreakthrough2(20)),
                //new EngineMCTS2(new MCTSBreakthrough(15))
                //new EngineAB(new HeuristicBreakthrough(), new ABBreakthrough())

                new EngineMCTS2(new MCTSGomoku(n -> n * n, 255)),
                //new EngineAB(new HeuristicGomoku(), new ABGomokuSortMoves())
                new EngineAB(new HeuristicGomokuNN("gomoku4MC10000.pt"), new ABGomokuSortMoves())
        );
        for (long time : TIMES_PER_TURN) {
            for (int startsEngine = 0; startsEngine < 2; startsEngine++) {
                int[] score = {0, 0, 0};
                int[] timeout = {0, 0};
                int[] illegalMove = {0, 0};
                long[] durations = {0, 0};
                int[] turns = {0, 0};
                long endTime = bean.getCurrentThreadCpuTime() + TIME_PER_PHASE;
                while (bean.getCurrentThreadCpuTime() < endTime) {

                    PositionGomoku position = new PositionGomoku(15);
                    //PositionBreakthrough position = new PositionBreakthrough(8);

                    int pl = startsEngine;
                    while (position.state() == GameState.ONGOING) {
                        long startTime = bean.getCurrentThreadCpuTime();
                        Object move = engines.get(pl).choseMove(position.copy(), startTime + time);
                        long duration = bean.getCurrentThreadCpuTime() - startTime;

                        turns[pl]++;
                        durations[pl] += duration;

                        if (duration > time) {
                            timeout[pl]++;
                        }
                        if (!position.moves().contains(move)) {
                            illegalMove[pl]++;
                            break;
                        }
                        position.applyMove(move);
                        pl = 1 - pl;
                        if (time > 60*SEC)
                            System.out.println("Continues position:\n" + position);
                    }
                    int id = (position.state() == GameState.DRAW) ? 2 : (1 - pl);
                    score[id]++;
                    //System.out.println("Position:\n" + position);
                }

                System.out.println("Time per turn: " + sec(time));
                System.out.println("Starts player: " + startsEngine);
                for (int i = 0; i < 2; i++) {
                    System.out.println(score[i]);
                    System.out.println(turns[i]);
                    System.out.println(sec(durations[i]));
                    if (engines.get(i) instanceof EngineMCTS2) {
                        System.out.println(((EngineMCTS2) engines.get(i)).count);
                        ((EngineMCTS2) engines.get(i)).count = 0;
                    }
                    if (engines.get(i) instanceof EngineAB) {
                        System.out.println(((EngineAB) engines.get(i)).visitedNodes);
                        ((EngineAB) engines.get(i)).visitedNodes = 0;
                    }
                }

                if (score[2] != 0)
                    System.out.println("Draws: " + score[2]);
                if (timeout[0] != 0 || timeout[1] != 0)
                    System.out.println("Timeout: " + timeout[0] + ":" + timeout[1]);
                if (illegalMove[0] != 0 || illegalMove[1] != 0)
                    System.out.println("IllegalMove: " + illegalMove[0] + ":" + illegalMove[1]);
            }
        }
    }
}
