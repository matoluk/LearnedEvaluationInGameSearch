import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Paths;

public class HeuristicGomokuNN implements PositionEvaluator {

    private final Model model;
    private final Predictor<float[], Float> predictor;
    private static final int SIZE = 15;
    private static final int INPUT_SIZE = SIZE * SIZE;
    private final float[] inputBuffer = new float[INPUT_SIZE];

    public HeuristicGomokuNN(String modelPath){

        model = Model.newInstance(modelPath);
        try {
            model.load(Paths.get(modelPath));
        } catch (Exception e) {
            e.printStackTrace();
        }

        predictor = model.newPredictor(new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext ctx, float[] input) {
                return new NDList(
                        ctx.getNDManager()
                                .create(input, new Shape(1, INPUT_SIZE))
                );
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList output) {
                return output.singletonOrThrow().getFloat();
            }
            @Override
            public Batchifier getBatchifier() {
                return null;
            }
        });
    }

    @Override
    public double eval(Position position) {
        if (!(position instanceof PositionGomoku pos))
            throw new IllegalArgumentException("Expected PositionGomoku");

        fillInputBuffer(pos);

        try {
            return -predictor.predict(inputBuffer);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    private void fillInputBuffer(PositionGomoku pos) {
        int idx = 0;

        for (PositionGomoku.Move m = new PositionGomoku.Move(); m != null; m = m.next(SIZE)) {
            int v = PositionGomoku.get(pos.board, m);
            inputBuffer[idx++] =
                    v == 0 ? 0f :
                            v == pos.actualPlayer() ? 1f : -1f;
        }
    }

    public void close() {
        predictor.close();
        model.close();
    }
}
