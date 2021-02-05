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

	/**
	 * Matrix constructor - need to provide number of rows and columns that matrix
	 * will have. Matrix elements are stored in array called data which dimensions
	 * are specified by provided rows and columns
	 * 
	 */
	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = new double[rows][columns];
	}

	/**
	 * Converts two-dimensional array arr into matrix
	 * 
	 * @param arr array to convert
	 * @return matrix two-dimensional array converted to matrix
	 * 
	 */
	public static Matrix toMatrix(double[][] arr) {
		Matrix matrix = new Matrix(arr.length, arr[0].length);
		matrix.data = arr;
		return matrix;
	}

	/**
	 * Converts one-dimensional array arr into matrix
	 * 
	 * @param arr array to convert
	 * @return matrix one-dimensional array converted to matrix
	 * 
	 */
	public static Matrix toMatrix(double[] arr) {
		Matrix matrix = new Matrix(1, arr.length);
		matrix.data[0] = arr;
		return matrix;
	}

	/**
	 * Adds scalar value to each element in Matrix a
	 * 
	 * @param scalar value that is added to elements of matrix a
	 * @param a      matrix to which scalar is added
	 * @return a original matrix with updated elements
	 * 
	 */
	public static Matrix add(double scalar, Matrix a) {
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				a.data[i][j] += scalar;
			}
		}
		return a;
	}

	/**
	 * Adds elements of matrix a to corresponding elements of matrix b
	 * 
	 * @param a first matrix
	 * @param b second matrix
	 * @return temp matrix of sums of corresponding elements from matrices a and b
	 * 
	 */
	public static Matrix add(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix temp = new Matrix(a.rows, a.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.columns; j++) {
					temp.data[i][j] = a.data[i][j] + b.data[i][j];
				}
			}
			return temp;
		} else {
			System.out.println("INCORRECT DIMENSIONS IN MATRIX ADDITION");
			return null;
		}
	}

	/**
	 * Subtracts scalar value from each element in Matrix a
	 * 
	 * @param scalar value that is subtracted from matrix a
	 * @param a      matrix from which scalar is subtracted
	 * @return a original matrix with updated elements
	 * 
	 */
	public static Matrix subtract(double scalar, Matrix a) {
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				a.data[i][j] -= scalar;
			}
		}
		return a;
	}

	/**
	 * Subtracts elements of matrix b from corresponding elements of matrix a
	 * 
	 * @param a first matrix
	 * @param b second matrix
	 * @return temp matrix of differences between corresponding elements from
	 *         matrices a and b
	 * 
	 */
	public static Matrix subtract(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix temp = new Matrix(a.rows, a.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.columns; j++) {
					temp.data[i][j] = a.data[i][j] - b.data[i][j];
				}
			}
			return temp;
		} else {
			System.out.println("INCORRECT DIMENSIONS IN MATRIX SUBTRACTION");
			return null;
		}
	}

	/**
	 * Multiplies each element of matrix a by a scalar value
	 * 
	 * @param scalar value by which elements of matrix a are multiplied
	 * @param a      matrix which elements are multiplied
	 * @return a original matrix with updated elements
	 * 
	 */
	public static Matrix multiply(double scalar, Matrix a) {
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				a.data[i][j] = a.data[i][j] * scalar;
			}
		}
		return a;
	}

	/**
	 * Multiplies elements of matrix a by corresponding elements of matrix b
	 * (element-wise matrix multiplicaton)
	 * 
	 * @param a first matrix
	 * @param b second matrix
	 * @return temp resulting matrix of the element-wise multipilication of matrices
	 *         a and b
	 * 
	 */
	public static Matrix multiply(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.columns == b.columns) {
			Matrix temp = new Matrix(a.rows, b.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < a.columns; j++) {
					temp.data[i][j] = a.data[i][j] * b.data[i][j];
				}
			}
			return temp;
		} else {
			System.out.println("INCORRECT DIMENSIONS IN MATRIX ELEMENT-WISE MULTIPLICATION");
			return null;
		}
	}

	/**
	 * Multiplies two matrices
	 * 
	 * @param a first matrix
	 * @param b second matrix
	 * @return temp resulting matrix which elements are dot products of rows from
	 *         matrix a and columns from matrix b
	 * 
	 */
	public static Matrix dot(Matrix a, Matrix b) {
		if (a.columns == b.rows) {
			Matrix temp = new Matrix(a.rows, b.columns);
			for (int i = 0; i < a.rows; i++) {
				for (int j = 0; j < b.columns; j++) {
					for (int k = 0; k < a.columns; k++) {
						temp.data[i][j] += a.data[i][k] * b.data[k][j];
					}
				}
			}
			return temp;
		} else {
			System.out.println("INCORRECT DIMENSIONS IN MATRIX MULTIPLICATION");
			return null;
		}
	}

	/**
	 * Transposes matrix to which this method is applied
	 * 
	 * @return matrix with transposed elements of an original matrix
	 * 
	 */
	public Matrix transpose() {
		Matrix temp = new Matrix(this.columns, this.rows);
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				temp.data[j][i] = this.data[i][j];
			}
		}
		return temp;
	}

	/**
	 * Computes logistic function output for each element of matrix a
	 * 
	 * @param a matrix which elements are put into logistic function
	 * @return temp resulting matrix which elements are outputs of logistic function
	 * 
	 */
	public static Matrix sigmoid(Matrix a) {
		Matrix temp = new Matrix(a.rows, a.columns);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp.data[i][j] = 1 / (1 + Math.exp(-a.data[i][j]));
			}
		}
		return temp;
	}

	/**
	 * Computes the natural logarithm of each element in matrix a
	 * 
	 * @param a matrix for which elements natural logarithm is computed
	 * @return temp resulting matrix which elements are natural logarithms of
	 *         elements from matrix a
	 * 
	 */
	public static Matrix log(Matrix a) {
		Matrix temp = new Matrix(a.rows, a.columns);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp.data[i][j] = Math.log(a.data[i][j]);
			}
		}
		return temp;
	}

	/**
	 * Sums up all elements in matrix a
	 * 
	 * @param a matrix which elements are summed up
	 * @return temp double value which is a sum of all elements in matrix a
	 * 
	 */
	public static double sum(Matrix a) {
		double temp = 0;
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp += a.data[i][j];
			}
		}
		return temp;
	}

	/**
	 * Sums up all elements in matrix a by rows or by columns
	 * 
	 * @param a    matrix which elements are summed up
	 * @param axis axis along which a sum is performed. If axis = 0 find sum of each
	 *             column. If axis = 1 find sum of each rows
	 * @return temp row vector with sum of each column if axis = 0 or column vector
	 *         with sum of each row if axis = 1
	 * 
	 */
	public static Matrix sum(Matrix a, int axis) {
		// sum values in each column
		if (axis == 0) {

			Matrix temp = new Matrix(1, a.columns);
			double sumCol;

			for (int i = 0; i < a.columns; i++) {
				sumCol = 0;
				for (int j = 0; j < a.rows; j++) {
					sumCol += a.data[j][i];
				}
				temp.data[0][i] = sumCol;
			}
			return temp;

			// sum values in each row
		} else if (axis == 1) {

			Matrix temp = new Matrix(a.rows, 1);
			double sumRow;

			for (int i = 0; i < a.rows; i++) {
				sumRow = 0;
				for (int j = 0; j < a.columns; j++) {
					sumRow += a.data[i][j];
				}
				temp.data[i][0] = sumRow;
			}
			return temp;
		} else {
			return null;
		}
	}

}
