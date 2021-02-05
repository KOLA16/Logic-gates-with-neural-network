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

	// Parametres
	private Matrix W1, b1, W2, b2;

	public NeuralNetwork(int inputSize, int hiddenLayerSize, int outputLayerSize) {
		this.W1 = new Matrix(hiddenLayerSize, inputSize);
		this.b1 = new Matrix(hiddenLayerSize, 1); // initialized to 0
		this.W2 = new Matrix(outputLayerSize, hiddenLayerSize);
		this.b2 = new Matrix(outputLayerSize, 1); // initialized to 0
	}

	private void initializeParameters() {
		// Initialize parameters W1 and W2 to random small values
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

		// b1 and b2 parameters are left initialized to 0
		// as this doesn't affect learning
	}

	private Map<String, Matrix> forwardPropagation(Matrix X) {
		// Add m-1 (m = number of training examples) columns to parameters b1 and b2
		// and fill them with values copied from the original column
		// It allows us to broadcast b1 and b2 across larger Matrices and perform
		// vectorized forward propagation on all the training examples
		Matrix b1m = new Matrix(this.b1.rows, X.columns);
		Matrix b2m = new Matrix(this.b2.rows, X.columns);

		for (int i = 0; i < b1m.rows; i++) {
			for (int j = 0; j < b1m.columns; j++) {
				b1m.data[i][j] = this.b1.data[i][0];
			}
		}

		for (int i = 0; i < b2m.rows; i++) {
			for (int j = 0; j < b2m.columns; j++) {
				b2m.data[i][j] = this.b2.data[i][0];
			}
		}

		// Implement forward propagation
		// Z1 = W1 o X + b1m
		Matrix Z1 = Matrix.add(Matrix.dot(this.W1, X), b1m);
		// A1 = sig(Z1)
		Matrix A1 = Matrix.sigmoid(Z1);
		// Z2 = W2 o A1 + b2m
		Matrix Z2 = Matrix.add(Matrix.dot(this.W2, A1), b2m);
		// A2 = sig(Z2)
		Matrix A2 = Matrix.sigmoid(Z2);

		Map<String, Matrix> cache = new HashMap<>();

		cache.put("Z1", Z1);
		cache.put("A1", A1);
		cache.put("Z2", A2);
		cache.put("A2", A2);

		return cache;
	}

	private double computeCost(Matrix A2, Matrix Y) {
		// number of examples
		int m = Y.columns;

		// create matrix where each element = 1 (for further computations)
		Matrix ones = new Matrix(Y.rows, Y.columns);
		for (int i = 0; i < ones.rows; i++) {
			for (int j = 0; j < ones.columns; j++) {
				ones.data[i][j] = 1;
			}
		}

		// logprob1 = log(A2) * Y
		Matrix logprob1 = Matrix.multiply(Matrix.log(A2), Y);
		// logprob2 = (1 - Y) * log(1 - A2)
		Matrix logprob2 = Matrix.multiply(Matrix.subtract(ones, Y), Matrix.log(Matrix.subtract(ones, A2)));
		// cost = -(1/m) * sum(log(A2) * Y + (1 - Y) * log(1 - A2)
		double cost = -(1.0 / m) * Matrix.sum(Matrix.add(logprob1, logprob2));

		return cost;
	}

	private Map<String, Matrix> backwardPropagation(Map<String, Matrix> cache, Matrix X, Matrix Y) {
		// number of examples
		int m = X.columns;

		Matrix A1 = cache.get("A1");
		Matrix A2 = cache.get("A2");

		// create matrix where each element = 1 (for further computations)
		Matrix ones = new Matrix(A1.rows, A1.columns);
		for (int i = 0; i < ones.rows; i++) {
			for (int j = 0; j < ones.columns; j++) {
				ones.data[i][j] = 1;
			}
		}

		// dZ2 = A2 - Y
		Matrix dZ2 = Matrix.subtract(A2, Y);
		// dW2 = (1/m) * dZ2 o A1.T
		Matrix dW2 = Matrix.multiply(1.0 / m, Matrix.dot(dZ2, A1.transpose()));
		// db2 = (1/m) * columnVectorThatContainsSumOfEachRow(dZ2)
		Matrix db2 = Matrix.multiply(1.0 / m, Matrix.sum(dZ2, 1));
		// dZ1 = W2.T o dZ2 * A2 * (1 - A2)
		Matrix dZ1 = Matrix.multiply(Matrix.dot(W2.transpose(), dZ2), Matrix.multiply(A1, Matrix.subtract(ones, A1)));
		// dW1 = (1/m) * dZ1 o X.T
		Matrix dW1 = Matrix.multiply(1.0 / m, Matrix.dot(dZ1, X.transpose()));
		// db1 = (1/m) * columnVectorThatContainsSumOfEachRow(dZ1)
		Matrix db1 = Matrix.multiply(1.0 / m, Matrix.sum(dZ1, 1));

		Map<String, Matrix> gradients = new HashMap<>();

		gradients.put("dW1", dW1);
		gradients.put("db1", db1);
		gradients.put("dW2", dW2);
		gradients.put("db2", db2);

		return gradients;
	}

	private void updateParameters(Map<String, Matrix> gradients, double learningRate) {

		Matrix dW1 = gradients.get("dW1");
		Matrix db1 = gradients.get("db1");
		Matrix dW2 = gradients.get("dW2");
		Matrix db2 = gradients.get("db2");

		// W1 = W1 - learningRate * dW1
		this.W1 = Matrix.subtract(this.W1, Matrix.multiply(learningRate, dW1));
		// b1 = b1 - learningRate * db1
		this.b1 = Matrix.subtract(this.b1, Matrix.multiply(learningRate, db1));
		// W2 = W2 - learningRate * dW2
		this.W2 = Matrix.subtract(this.W2, Matrix.multiply(learningRate, dW2));
		// b2 = b2 - learningRate * db2
		this.b2 = Matrix.subtract(this.b2, Matrix.multiply(learningRate, db2));
	}

	public void trainParameters(Matrix X, Matrix Y, int iterations, double learningRate) {

		// Initialize parameters
		this.initializeParameters();

		// Gradient descent
		for (int i = 0; i <= iterations; i++) {
			// Forward propagation
			Map<String, Matrix> cache = this.forwardPropagation(X);

			// Cost function
			double cost = this.computeCost(cache.get("A2"), Y);

			// Backpropagation
			Map<String, Matrix> gradients = this.backwardPropagation(cache, X, Y);

			// Parameters update
			this.updateParameters(gradients, learningRate);

			// Print cost
			if (i % 1000 == 0) {
				System.out.println("Cost after iteration " + i + ": " + cost);
			}
		}
	}

	public Map<String, Matrix> returnParameters() {
		Map<String, Matrix> parameters = new HashMap<>();

		parameters.put("W1", this.W1);
		parameters.put("b1", this.W1);
		parameters.put("W2", this.W1);
		parameters.put("b2", this.W1);

		return parameters;
	}

	public Matrix predict(Matrix X) {
		Map<String, Matrix> cache = this.forwardPropagation(X);

		Matrix predictions = cache.get("A2");
		return predictions;
	}
}
