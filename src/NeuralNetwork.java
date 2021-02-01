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

	// private Matrix X;
	// private Matrix Y;
	// private int hiddenLayerSize;

	public NeuralNetwork(Matrix X, Matrix Y, int hiddenLayerSize) {
		// this.X = X;
		// this.Y = Y;
		// this.hiddenLayerSize = hiddenLayerSize;
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

	public Map<String, Matrix> forwardPropagation(Matrix X, Map<String, Matrix> parameters) {
		// Retrive each parameter from Map 'parameters'
		Matrix W1 = parameters.get("W1");
		Matrix b1 = parameters.get("b1");
		Matrix W2 = parameters.get("W2");
		Matrix b2 = parameters.get("b2");

		// Add m-1 (m = number of training examples) columns to parameters b1 and b2
		// and fill them with values copied from the original column
		// It allows us to broadcast b1 and b2 across larger Matrices and perform
		// vectorized forward propagation on all the training examples
		Matrix b1m = new Matrix(b1.rows, X.columns);
		Matrix b2m = new Matrix(b2.rows, X.columns);

		for (int i = 0; i < b1m.rows; i++) {
			for (int j = 0; j < b1m.columns; j++) {
				b1m.data[i][j] = b1.data[i][0];
			}
		}

		for (int i = 0; i < b2m.rows; i++) {
			for (int j = 0; j < b2m.columns; j++) {
				b2m.data[i][j] = b2.data[i][0];
			}
		}

		// Implement forward propagation
		// Z1 = W1 o X + b1m
		Matrix Z1 = Matrix.add(Matrix.multiply(W1, X), b1m);
		// A1 = sig(Z1)
		Matrix A1 = Matrix.sigmoid(Z1);
		// Z2 = W2 o A1 + b2m
		Matrix Z2 = Matrix.add(Matrix.multiply(W2, A1), b2m);
		// A2 = sig(Z2)
		Matrix A2 = Matrix.sigmoid(Z2);

		Map<String, Matrix> cache = new HashMap<>();

		cache.put("Z1", Z1);
		cache.put("A1", A1);
		cache.put("Z2", A2);
		cache.put("A2", A2);

		return cache;
	}

	public double computeCost(Matrix A2, Matrix Y) {
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
		Matrix logprob1 = Matrix.multiplyElementWise(Matrix.log(A2), Y);
		// logprob2 = (1 - Y) * log(1 - A2)
		Matrix logprob2 = Matrix.multiplyElementWise(Matrix.subtract(ones, Y), Matrix.log(Matrix.subtract(ones, A2)));
		// cost = -(1/m) * sum(log(A2) * Y + (1 - Y) * log(1 - A2)
		double cost = -(1.0 / m) * Matrix.sum(Matrix.add(logprob1, logprob2));

		return cost;
	}

	public Map<String, Matrix> backwardPropagation(Map<String, Matrix> parameters, Map<String, Matrix> cache, Matrix X,
			Matrix Y) {
		// number of examples
		int m = X.columns;

		Matrix W1 = parameters.get("W1");
		Matrix W2 = parameters.get("W2");

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
		Matrix dW2 = Matrix.multiply(1.0 / m, Matrix.multiply(dZ2, Matrix.transpose(A1)));
		// db2 = (1/m) * columnVectorThatContainsSumOfEachRow(dZ2)
		Matrix db2 = Matrix.multiply(1.0 / m, Matrix.sum(dZ2, 1));
		// dZ1 = W2.T o dZ2 * A2 * (1 - A2)
		Matrix dZ1 = Matrix.multiplyElementWise(Matrix.multiply(Matrix.transpose(W2), dZ2),
				Matrix.multiplyElementWise(A1, Matrix.subtract(ones, A1)));
		// dW1 = (1/m) * dZ1 o X.T
		Matrix dW1 = Matrix.multiply(1.0 / m, Matrix.multiply(dZ1, Matrix.transpose(X)));
		// db1 = (1/m) * columnVectorThatContainsSumOfEachRow(dZ1)
		Matrix db1 = Matrix.multiply(1.0 / m, Matrix.sum(dZ1, 1));

		Map<String, Matrix> gradients = new HashMap<>();

		gradients.put("dW1", dW1);
		gradients.put("db1", db1);
		gradients.put("dW2", dW2);
		gradients.put("db2", db2);

		return gradients;
	}

	public Map<String, Matrix> updateParameters(Map<String, Matrix> parameters, Map<String, Matrix> gradients,
			double learningRate) {

		Matrix W1 = parameters.get("W1");
		Matrix b1 = parameters.get("b1");
		Matrix W2 = parameters.get("W2");
		Matrix b2 = parameters.get("b2");

		Matrix dW1 = gradients.get("dW1");
		Matrix db1 = gradients.get("db1");
		Matrix dW2 = gradients.get("dW2");
		Matrix db2 = gradients.get("db2");

		// W1 = W1 - learningRate * dW1
		W1 = Matrix.subtract(W1, Matrix.multiply(learningRate, dW1));
		// b1 = b1 - learningRate * db1
		b1 = Matrix.subtract(b1, Matrix.multiply(learningRate, db1));
		// W2 = W2 - learningRate * dW2
		W2 = Matrix.subtract(W2, Matrix.multiply(learningRate, dW2));
		// b2 = b2 - learningRate * db2
		b2 = Matrix.subtract(b2, Matrix.multiply(learningRate, db2));

		parameters.put("W1", W1);
		parameters.put("b1", b1);
		parameters.put("W2", W2);
		parameters.put("b2", b2);

		return parameters;
	}

	public Map<String, Matrix> trainParameters(Matrix X, Matrix Y, int hiddenLayerSize, int iterations,
			double learningRate) {
		// Input and output layer sizes
		int nX = X.rows;
		int nY = Y.rows;

		// Initialize parameters
		Map<String, Matrix> parameters = this.initializeParameters(nX, hiddenLayerSize, nY);

		// Gradient descent
		for (int i = 0; i <= iterations; i++) {
			// Forward propagation
			Map<String, Matrix> cache = this.forwardPropagation(X, parameters);

			// Cost function
			double cost = this.computeCost(cache.get("A2"), Y);

			// Backpropagation
			Map<String, Matrix> gradients = this.backwardPropagation(parameters, cache, X, Y);

			// Parameters update
			parameters = this.updateParameters(parameters, gradients, learningRate);

			// Print cost
			if (i % 1000 == 0) {
				System.out.println("Cost after iteration " + i + ": " + cost);
			}
		}

		return parameters;
	}
	
	public Matrix predict(Map<String, Matrix> parameters, Matrix X) {
		Map<String, Matrix> cache = this.forwardPropagation(X, parameters);
		
		Matrix predictions = cache.get("A2");
		return predictions;
	}

	
	// FOR TESTING PURPOSES ONLY
	public static void main(String[] args) {

		Matrix X = new Matrix(2, 4);
		Matrix Y = new Matrix(1, 4);

		X.data[0][0] = 1.0;
		X.data[1][0] = 1.0;
		X.data[0][1] = 0.0;
		X.data[1][1] = 0.0;
		X.data[0][2] = 1.0;
		X.data[1][2] = 0.0;
		X.data[0][3] = 0.0;
		X.data[1][3] = 1.0;

		Y.data[0][0] = 1.0;
		Y.data[0][1] = 1.0;
		Y.data[0][2] = 0.0;
		Y.data[0][3] = 0.0;

		NeuralNetwork nn = new NeuralNetwork(X, Y, 2);
		Map<String, Matrix> parameters = nn.trainParameters(X, Y, 2, 2000000, 5.0);

		for (int i = 0; i < parameters.get("b1").rows; i++) {
			System.out.println(parameters.get("b1").data[i][0]);
		}

		for (int i = 0; i < parameters.get("W1").rows; i++) {
			for (int j = 0; j < parameters.get("W1").columns; j++)
				System.out.println(parameters.get("W1").data[i][j]);
		}
		
		Matrix predictions = nn.predict(parameters, X);
		
		for (int i = 0; i < predictions.rows; i++) {
			for (int j = 0; j < predictions.columns; j++) {
				System.out.println(predictions.data[i][j]);
			}
		}
	}

}
