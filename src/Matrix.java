/**
 * Matrix.java
 * 
 * A class that can be used to create matrices and perform basic matrix
 * operations
 *
 */

public class Matrix {

	int rows, columns;
	double[][] data;

	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = new double[rows][columns];
	}

	public static Matrix toMatrix(double[][] arr) {
		Matrix matrix = new Matrix(arr.length, arr[0].length);
		matrix.data = arr;
		return matrix;
	}
	
	public static Matrix toMatrix(double[] arr) {
		Matrix matrix = new Matrix(1, arr.length);
		matrix.data[0] = arr;
		return matrix;
	}

	public void add(double scalar) {
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.data[i][j] += scalar;
			}
		}
	}

	public static Matrix add(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix sum = new Matrix(a.rows, a.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.columns; j++) {
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
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.data[i][j] -= scalar;
			}
		}
	}

	public static Matrix subtract(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix difference = new Matrix(a.rows, a.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.columns; j++) {
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
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.data[i][j] *= scalar;
			}
		}
	}

	public static Matrix multiply(Matrix a, Matrix b) {
		if (a.columns == b.rows) {
			Matrix product = new Matrix(a.rows, b.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < b.columns; j++) {
					for (int k = 0; k < a.columns; k++) {
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
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.columns; j++) {
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
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				temp.data[j][i] = this.data[i][j];
			}
		}
		this.data = temp.data;
	}

	public static double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	public static Matrix sigmoid(Matrix a) {
		Matrix temp = new Matrix(a.rows, a.columns);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp.data[i][j] = 1 / (1 + Math.exp(-a.data[i][j]));
			}
		}
		return temp;
	}

}
