/**
 * Driver.java
 * 
 * Class that initializes required matrices, hyperparameters,
 * and runs learning
 *
 */

public class Driver {

		public static void main(String[] args) {

			double[][] input = {{1.0, 0.0, 1.0, 0.0}, 
					            {1.0, 0.0, 0.0, 1.0}};
			
			double[] output = {1.0, 1.0, 0.0, 0.0}; 
		            
			Matrix X = Matrix.toMatrix(input);
			Matrix Y = Matrix.toMatrix(output);
			int hiddenUnits = 5;
			int iterations = 50000;
			double learningRate = 4.0;
					
			NeuralNetwork nn = new NeuralNetwork(X.rows, hiddenUnits, Y.rows);
			nn.trainParameters(X, Y, iterations, learningRate);

			double[][] testInput = {{1.0, 0.0, 0.0, 1.0, 1.0, 0.0}, 
					                {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}};
			
			Matrix Xtest = Matrix.toMatrix(testInput);
			nn.predict(Xtest);
		}
}
