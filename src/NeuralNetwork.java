/**
 * NeuralNetwork.java
 * 
 * A class to represent a neural network
 *  
 *
 */

public class NeuralNetwork {

    private Matrix input;
	private Matrix[] weights;
	private int hiddenLayers;

	public NeuralNetwork(Matrix input, Matrix[] weights, int hiddenLayers) {
		this.input = input;
		this.weights = weights;
		this.hiddenLayers = hiddenLayers;
	}

	public double[][] forwardPropagation() {
		Matrix activations = this.input;
		for (int i = 0; i <= this.hiddenLayers; i++) {
			Matrix z = Matrix.multiply(this.weights[i], activations);
			// Add a bias unit for the pre-output layer
			if (i < this.hiddenLayers) {  
				activations = new Matrix(weights[i].rows + 1, 1);
				activations.data[0][0] = 1; 
				// Copy a sigmoid matrix to activations matrix 
				// activations.data[0][0] = 1 (bias unit),
				// activations.data[1...data.rows-1][0] = sigmoid value for each unit in the layer
				System.arraycopy(Matrix.sigmoid(z).data, 0, activations.data, 1, Matrix.sigmoid(z).rows); 																							
			} else {
				activations = Matrix.sigmoid(z);
			}

		}
		return activations.data;
	}

}
