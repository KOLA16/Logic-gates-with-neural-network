/**
 * Matrix.java
 * 
 * A class that can be used to create matrices and perform basic matrix
 * operations
 *
 */

public class Matrix {

	private int rows;
	private int columns;
	private double[][] data;

	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = new double[rows][columns];
	}

	public void add(double scalar) {
		for (int i = 0; i <= this.rows - 1; i++) {
			for (int j = 0; j <= this.columns - 1; j++) {
				this.data[i][j] += scalar;
			}
		}
    }

	public static Matrix add(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix sum = new Matrix(a.rows, a.columns);
			for (int i = 0; i <= a.rows - 1; i++) {
				for (int j = 0; j <= a.columns - 1; j++) {
					sum.data[i][j] = a.data[i][j] + b.data[i][j];
				}
			}
			return sum;
		} else {
			System.out.println("INCORRECT DIMENSIONS");
			return null;
		}
	}

}
