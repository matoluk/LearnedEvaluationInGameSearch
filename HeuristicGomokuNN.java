import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.file.Paths;

public class HeuristicGomokuNN implements PositionEvaluator {

    private final Model model;
    private final Predictor<float[], Float> predictor;
    private final int inputSize = 15 * 15;

    public HeuristicGomokuNN(String modelPath){

        model = Model.newInstance(modelPath);
        try {
            model.load(Paths.get(modelPath));
        } catch (Exception e) {
            e.printStackTrace();
        }

        predictor = model.newPredictor(new Translator<float[], Float>() {
            @Override
            public NDList processInput(TranslatorContext ctx, float[] input) {
                NDManager manager = ctx.getNDManager();
                return new NDList(manager.create(input).reshape(1, input.length));
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

        String[] tokens = pos.flattenBoard().split(",");
        float[] input = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            input[i] = Float.parseFloat(tokens[i]);
        }

        try {
            return predictor.predict(input);
        } catch (TranslateException e) {
            e.printStackTrace();
            return 0.0;
        }
    }

    public void close() {
        predictor.close();
        model.close();
    }
}
