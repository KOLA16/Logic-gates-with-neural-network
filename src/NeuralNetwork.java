import java.util.HashMap;
import java.util.Map;

/**
 * NeuralNetwork.java
 * 
 * A class that represents two layer neural network (one hidden layer), and
 * provides methods to perform parameters learning.
 *
 */

public class NeuralNetwork {

	// Parametres
	private Matrix W1, b1, W2, b2;

	/**
	 * NeuralNetwork constructor - need to provide input layer size, hidden layer
	 * size and output layer size. Creates parameters matrices based on provided
	 * dimensions
	 * 
	 */
	public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
		this.W1 = new Matrix(hiddenLayerSize, inputLayerSize);
		this.b1 = new Matrix(hiddenLayerSize, 1); 
		this.W2 = new Matrix(outputLayerSize, hiddenLayerSize);
		this.b2 = new Matrix(outputLayerSize, 1); 
	}

	/**
	 * Initializes weights matrices W1 and W2 to random small values
	 * 
	 */
	private void initializeParameters() {
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

		// b1 and b2 bias vectors are left initialized to 0
		// as this doesn't affect learning
	}

	/**
	 * Performs forward propagation step. Computes the inputs of the activation
	 * function (Z1, Z2), and the sigmoid output of the activations in the hidden and
	 * the output layer (A1, A2). It caches A1 and A2 for backward propagation step.
	 * 
	 * @param X input matrix of dimensions: (inputLayerSize x number of training
	 *          examples). Each column stores features of a single training example.
	 * @return cache Map that stores sigmoid outputs of the activations: A1 and A2.
	 * 
	 */
	private Map<String, Matrix> forwardPropagation(Matrix X) {
		// Add m-1 (m = number of training examples) columns to bias vectors b1 and b2
		// and fill them with values copied from the original vector
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
		cache.put("A1", A1);
		cache.put("A2", A2);

		return cache;
	}

	/**
	 * Computes the cross-entropy cost
	 * 
	 * @param A2 sigmoid output matrix of the last layer of dimensions:
	 *           (outputLayerSize x number of training examples). Each column stores
	 *           output for a single training example.
	 * @param Y  label matrix of dimensions: (outputLayerSize x number of training
	 *           examples) Each column labels a single training example as 1 or 0
	 * @return cost cross-entropy cost
	 * 
	 */
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

	/**
	 * Performs backward propagation step. Computes gradients of parameters.
	 * 
	 * @param cache Map that stores sigmoid outputs of the activations: A1 and A2.
	 *              It is passed from the forward propagation function.
	 * @param X     input matrix of dimensions: (inputLayerSize x number of training
	 *              examples). Each column stores features of a single training
	 *              example.
	 * @param Y     label matrix of dimensions: (outputLayerSize x number of
	 *              training examples) Each column labels a single training example
	 *              as 1 or 0
	 * @return gradients Map that stores gradients of parameters W1, b1, W2, and b2
	 * 
	 */
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

	/**
	 * Updates weights and bias terms. Gradient descent step.
	 * 
	 * @param gradients    Map that stores gradients of parameters W1, b1, W2, and
	 *                     b2
	 * @param learningRate controls how quickly the model is learning
	 * 
	 */
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

	/**
	 * Neural network model. Calls all the required methods to perform parameters
	 * training.
	 * 
	 * @param X            input matrix of dimensions: (inputLayerSize x number of
	 *                     training examples). Each column stores features of a
	 *                     single training example.
	 * @param Y            label matrix of dimensions: (outputLayerSize x number of
	 *                     training examples) Each column labels a single training
	 *                     example as 1 or 0.
	 * @param iterations   number of iterations in the gradient descent loop
	 * @param learningRate controls how quickly the model is learning
	 * 
	 */
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

		System.out.println("\nLearned parameters:\nW1 =\n" + this.W1.toString() + "\nb1 =\n" + this.b1.toString()
				+ "\nW2 =\n" + this.W2.toString() + "\nb2 =\n" + this.b2.toString());
	}

	/**
	 * Using the learned parameters, predicts a label for each example in matrix X
	 * 
	 * @param X input matrix of dimensions: (inputLayerSize x number of training
	 *          examples). Each column stores features of a single training example.
	 * 
	 */
	public void predict(Matrix X) {
		Map<String, Matrix> cache = this.forwardPropagation(X);
		Matrix outputs = cache.get("A2");

		Matrix predictions = new Matrix(outputs.rows, outputs.columns);
		for (int i = 0; i < predictions.rows; i++) {
			for (int j = 0; j < predictions.columns; j++) {
				predictions.data[i][j] = Math.round(outputs.data[i][j]);
			}
		}

		System.out.println("Output layer activations (probability that output = 1): " + outputs.toString());
		System.out.println("Predictions: " + predictions.toString());
	}
}
