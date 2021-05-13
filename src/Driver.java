/**
 * Driver.java
 * 
 * Class that initializes required matrices, hyperparameters,
 * and runs learning
 *
 */

public class Driver {

		public static void main(String[] args) {
            // Initialize hyperparameters
			final int hiddenUnits = 5;
			final int iterations = 10000;
			final double learningRate = 4.0;
			
			// Truth tables
			final double[][] input = {{1.0, 0.0, 1.0, 0.0}, 
					            	  {1.0, 0.0, 0.0, 1.0}};
			    
			final double[] xnorLabels = {1.0, 1.0, 0.0, 0.0};
			final double[] xorLabels = {0.0, 0.0, 1.0, 1.0};
			final double[] norLabels = {0.0, 1.0, 0.0, 0.0};
			final double[] nandLabels = {0.0, 1.0, 1.0, 1.0};
			
			Matrix X = Matrix.toMatrix(input);
			
			// ~~~~~~ XNOR gate ~~~~~~ 
			Matrix Y = Matrix.toMatrix(xnorLabels);
			
			System.out.println("PERFORM XNOR GATE LEARNING:\n");
			
			NeuralNetwork nn = new NeuralNetwork(X.rows, hiddenUnits, Y.rows);
			nn.trainParameters(X, Y, iterations, learningRate);
			
			System.out.println("Input:\n" + X.toString());
			nn.predict(X);
			
			/*// ~~~~~~ XOR gate ~~~~~~ 
		    Matrix Y = Matrix.toMatrix(xorLabels);
			
			System.out.println("PERFORM XOR GATE LEARNING:\n");
			
			NeuralNetwork nn = new NeuralNetwork(X.rows, hiddenUnits, Y.rows);
			nn.trainParameters(X, Y, iterations, learningRate);
			
			System.out.println("Input:\n" + X.toString());
			nn.predict(X);*/
			
			/*// ~~~~~~ NOR gate ~~~~~~ 
			Matrix Y = Matrix.toMatrix(norLabels);
			
			System.out.println("PERFORM NOR GATE LEARNING:\n");
			
			NeuralNetwork nn = new NeuralNetwork(X.rows, hiddenUnits, Y.rows);
			nn.trainParameters(X, Y, iterations, learningRate);
			
			System.out.println("Input:\n" + X.toString());
			nn.predict(X);*/
			
			/*// ~~~~~~ NAND gate ~~~~~~ 
			Matrix Y = Matrix.toMatrix(nandLabels);
			
			System.out.println("PERFORM NAND GATE LEARNING:\n");
			
			NeuralNetwork nn = new NeuralNetwork(X.rows, hiddenUnits, Y.rows);
			nn.trainParameters(X, Y, iterations, learningRate);
			
			System.out.println("Input:\n" + X.toString());
			nn.predict(X);*/
		}
}
