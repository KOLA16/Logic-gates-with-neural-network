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

	public Matrix add(double scalar) {
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.data[i][j] += scalar;
			}
		}
		return this;
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

	public static Matrix multiply(double scalar, Matrix a) {
		Matrix temp = new Matrix(a.rows, a.columns);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp.data[i][j] = a.data[i][j] * scalar;
			}
		}
		return temp;
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

	public Matrix transpose() {
		Matrix temp = new Matrix(this.columns, this.rows);
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				temp.data[j][i] = this.data[i][j];
			}
		}
		return temp;
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
	
	public static Matrix log(Matrix a) {
		Matrix temp = new Matrix(a.rows, a.columns);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp.data[i][j] = Math.log(a.data[i][j]);
			}
		}
		return temp;
	}
	
	public static double sum(Matrix a) {
		double temp = 0;
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.columns; j++) {
				temp += a.data[i][j];
			}
		}
		return temp;
	}
	
	public static Matrix sum(Matrix a, int axis) {
		Matrix temp = new Matrix(1,1);
		
		if (axis == 0) { // sum values in each column
			
			temp = new Matrix(1, a.columns);
			double sumCol;
			
			for(int i = 0; i < a.columns; i++){ 
				sumCol = 0;
	            for(int j = 0; j < a.rows; j++){  
	                sumCol += a.data[j][i];  
	            }  
	            temp.data[0][i] = sumCol; 
	        } 
			
		} else if (axis == 1) { // sum values in each row
			
			temp = new Matrix(a.rows, 1);
			double sumRow;
			
			for(int i = 0; i < a.rows; i++){  
				sumRow = 0;
		        for(int j = 0; j < a.columns; j++){  
		            sumRow += a.data[i][j];  
		        }  
		        temp.data[i][0] = sumRow; 
		    }  
		}
		
		return temp;
	}

}
