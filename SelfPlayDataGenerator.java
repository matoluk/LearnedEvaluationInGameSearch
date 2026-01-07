import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;

public class SelfPlayDataGenerator {

    private static final long TIME_GRANULARITY = 15_625_000;
    private static final long TIME_PER_TURN = 128 * TIME_GRANULARITY; // 64 -> 1 sec
    private static final ThreadMXBean bean = ManagementFactory.getThreadMXBean();

    private final Engine[] engines;
    private final Position startPosition;
    private final int gamesToPlay;
    private final String outputFile;

    public SelfPlayDataGenerator(Engine engine1, Engine engine2, Position position, int gamesToPlay, String outputFile) {
        engines = new Engine[]{engine1, engine2};
        startPosition = position;
        this.gamesToPlay = gamesToPlay;
        this.outputFile = outputFile;
    }

    public void generate() {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {

            for (int g = 0; g < gamesToPlay; g++) {
                Position position = startPosition.copy();
                int plOnTurn = 0;

                List<String> positions = new ArrayList<>();

                while (position.state() == GameState.ONGOING) {
                    long startTime = bean.getCurrentThreadCpuTime();
                    Object move = engines[plOnTurn].choseMove(position.copy(), startTime + TIME_PER_TURN);
                    long duration = bean.getCurrentThreadCpuTime() - startTime;

                    if (!position.moves().contains(move) || duration > TIME_PER_TURN) {
                        System.out.println("Illegal move or timeout. Ending game early.");
                        break;
                    }

                    position.applyMove(move);
                    plOnTurn = 1 - plOnTurn;

                    positions.add(position.flattenBoard());
                }

                int result = (position.state() == GameState.DRAW) ? 0 : (1 - 2 * plOnTurn);

                for (String pos : positions) {
                    writer.write(pos + result + "\n");
                    result *= -1;
                }

                System.out.println("Generated game " + (g + 1) + "/" + gamesToPlay + ", result: " + ((positions.size() % 2)*2-1));
            }

            System.out.println("Data generation finished. Saved to " + outputFile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        Engine engine1 = new EngineAB(new HeuristicGomoku(), new ABGomokuSortMoves());
        Engine engine2 = new EngineAB(new HeuristicGomoku(), new ABGomokuSortMoves());
        //Engine engine1 = new EngineMCTS2(new MCTSGomoku(n -> n * n, 30));
        //Engine engine2 = new EngineMCTS2(new MCTSGomoku(n -> n * n, 30));
        Position position = new PositionGomoku(15);

        /*
        Engine engine1 = new EngineAB(new HeuristicBreakthrough(), new ABBreakthrough());
        Engine engine2 = new EngineAB(new HeuristicBreakthrough(), new ABBreakthrough());
        Position position = new PositionBreakthrough(8);
        */

        SelfPlayDataGenerator generator = new SelfPlayDataGenerator(engine1, engine2, position, 200, "gomoku128AB200.csv");

        generator.generate();
    }
}
