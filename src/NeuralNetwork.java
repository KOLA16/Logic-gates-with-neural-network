import java.util.HashMap;
import java.util.Map;

/**
 * NeuralNetwork.java
 * 
 * A class to represent a neural network
 * 
 *
 */

public class NeuralNetwork {

	private Matrix X;
	private Matrix Y;
	private int hiddenLayerSize;

	public NeuralNetwork(Matrix X, Matrix Y, int hiddenLayerSize) {
		this.X = X;
		this.Y = Y;
		this.hiddenLayerSize = hiddenLayerSize;
	}

	public Map<String, Matrix> initializeParameters(int inputSize, int hiddenLayerSize, int outputLayerSize) {
		Matrix W1 = new Matrix(hiddenLayerSize, inputSize);
		Matrix b1 = new Matrix(hiddenLayerSize, 1); // initialized to 0
		Matrix W2 = new Matrix(outputLayerSize, hiddenLayerSize);
		Matrix b2 = new Matrix(outputLayerSize, 1); // initialized to 0

		// initialize parameters W1 and W2 to random small values
		for (int i = 0; i < W1.rows; i++) {
			for (int j = 0; j < W1.columns; j++) {
				W1.data[i][j] = Math.random() * 0.01;
			}
		}

		for (int i = 0; i < W2.rows; i++) {
			for (int j = 0; j < W2.columns; j++) {
				W2.data[i][j] = Math.random() * 0.01;
			}
		}

		Map<String, Matrix> parameters = new HashMap<>();

		parameters.put("W1", W1);
		parameters.put("b1", b1);
		parameters.put("W2", W2);
		parameters.put("b2", b2);

		return parameters;
	}

}
