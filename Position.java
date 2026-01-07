import java.io.Serializable;
import java.util.List;

public interface Position extends Serializable {
    Position copy();
    List<Object> moves();
    GameState state();
    Position applyMove(Object move);
    void revertMove(Object move);
    String flattenBoard();
}
