import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.IOException;
import java.util.Set;

public class HeuristicSpeedTest {

    static Object read(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName + ".dat"))) {
            return in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) {
        PositionEvaluator heuristic = new HeuristicGomokuNN("gomoku4MC10000.pt");
        //PositionEvaluator heuristic = new HeuristicGomoku();

        Set<PositionGomoku> positions = (Set<PositionGomoku>) read("setGomokuPositions");
        if (positions == null || positions.isEmpty()) {
            System.out.println("No positions loaded.");
            return;
        }
        System.out.println("Loaded positions: " + positions.size());


        // warm-up
        for (PositionGomoku p : positions) {
            heuristic.eval(p);
        }

        long start = System.nanoTime();
        for (PositionGomoku p : positions) {
            heuristic.eval(p);
        }
        long timeNs = System.nanoTime() - start;
        double timeMs = timeNs / 1_000_000.0;

        System.out.println("Total time: " + timeMs + " ms");
    }
}
