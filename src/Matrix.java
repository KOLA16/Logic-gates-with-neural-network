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
	
	public void subtract(double scalar) {
		for (int i = 0; i <= this.rows - 1; i++) {
			for (int j = 0; j <= this.columns - 1; j++) {
				this.data[i][j] -= scalar;
			}
		}
    }
	
	public static Matrix subtract(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix difference = new Matrix(a.rows, a.columns);
			for (int i = 0; i <= a.rows - 1; i++) {
				for (int j = 0; j <= a.columns - 1; j++) {
					difference.data[i][j] = a.data[i][j] - b.data[i][j];
				}
			}
			return difference;
		} else {
			System.out.println("INCORRECT DIMENSIONS");
			return null;
		}
	}
	
	public void multiply(double scalar) {
		for (int i = 0; i <= this.rows - 1; i++) {
			for (int j = 0; j <= this.columns - 1; j++) {
				this.data[i][j] *= scalar;
			}
		}
	}

	public static Matrix multiply(Matrix a, Matrix b) {
		if (a.columns == b.rows) {
			Matrix product = new Matrix(a.rows, b.columns);
			for (int i = 0; i <= a.rows - 1; i++) {
				for (int j = 0; j <= b.columns - 1; j++) {
					for (int k = 0; k <= a.columns - 1; k++) {
						product.data[i][j] += a.data[i][k] * b.data[k][j];
					}
				}
			}
			return product;
		} else {
			System.out.println("INCORRECT DIMENSIONS");
			return null;
		}
	}
	
	public static Matrix multiplyElementWise(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix product = new Matrix(a.rows, b.columns);
			for (int i = 0; i <= a.rows - 1; i++) {
				for (int j = 0; j <= a.columns - 1; j++) {
						product.data[i][j] = a.data[i][j] * b.data[i][j];
				}
			}
			return product;
		} else {
			System.out.println("INCORRECT DIMENSIONS");
			return null;
		}
	}
	
	public void transpose() {
		Matrix temp = new Matrix(this.columns, this.rows);
		for (int i=0; i<=this.rows-1; i++) {
			for (int j=0; j<=this.columns-1; j++) {
				temp.data[j][i] = this.data[i][j];
			}
		}
		this.data = temp.data;
	}
					
}
