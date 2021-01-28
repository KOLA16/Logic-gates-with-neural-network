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

	public Map<String, Matrix> initializeParameters(int inputSize, int hiddenLayerSize, int outputLayerSize, int m) {
		Matrix W1 = new Matrix(hiddenLayerSize, inputSize);
		Matrix b1 = new Matrix(hiddenLayerSize, m); // initialized to 0
		Matrix W2 = new Matrix(outputLayerSize, hiddenLayerSize);
		Matrix b2 = new Matrix(outputLayerSize, m); // initialized to 0

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

	public Map<String, Matrix> forwardPropagation(Matrix X, Map<String, Matrix> parameters) {
		// Retrive each parameter from Map 'parameters'
		Matrix W1 = parameters.get("W1");
		Matrix b1 = parameters.get("b1");
		Matrix W2 = parameters.get("W2");
		Matrix b2 = parameters.get("b2");

		// Implement forward propagation
		Matrix Z1 = Matrix.add(Matrix.multiply(W1, X), b1);
		Matrix A1 = Matrix.sigmoid(Z1);
		Matrix Z2 = Matrix.add(Matrix.multiply(W2, A1), b2);
		Matrix A2 = Matrix.sigmoid(Z2);

		Map<String, Matrix> cache = new HashMap<>();

		cache.put("Z1", Z1);
		cache.put("A1", A1);
		cache.put("Z2", A2);
		cache.put("A2", A2);

		return cache;
	}
	
	public double computeCost(Matrix A2, Matrix Y, Map<String, Matrix> parameters) {
		
		int m = Y.columns; // number of examples
		Matrix ones = new Matrix(Y.rows, Y.columns); 
		
		for (int i = 0; i < ones.rows; i++) {
			for (int j = 0; j < ones.columns; j++) {
				ones.data[i][j] = 1;
			}
		}
		
		Matrix logprob1 = Matrix.multiplyElementWise(Matrix.log(A2), Y); // log(A2) * Y
		Matrix logprob2 = Matrix.multiplyElementWise(Matrix.subtract(ones, Y), Matrix.log(Matrix.subtract(ones, A2))); // (1 - Y) * log(1 - A2)
		double cost = -(1/m) * Matrix.sum(Matrix.add(logprob1, logprob2));
		
		return cost;
	}

}
