
import java.util.LinkedList;
import java.util.Scanner;


class Matrix{
	
	int Row,Col;
	int[][] blocks;
	
	Matrix(int Row,int Col,int[][] blocks){
		this.Row = Row;
		this.Col = Col;
		this.blocks = blocks;
	}	
	

	
	static int[][] generate_blocks(int row,int col,String[] data) {
		
		int[][] blocks = new int[row][col];
		
		for(int i = 0; i < data.length; i++) {
			String eachRow = data[i];
			if(eachRow.length() < 5) {
				continue;
			}
			String[] items = eachRow.split(" ");
			int ROW = Integer.parseInt(items[0]);
			for(int j = 1; j < items.length; j++) {
				try{
					int COL = Integer.parseInt(items[j].substring(0,items[j].indexOf(":")).replaceAll(" ", ""));
					int DATA =  Integer.parseInt(items[j].substring(items[j].indexOf(":")+1).replaceAll(" ", ""));
					blocks[ROW-1][COL-1] = DATA;
				}
				catch(Exception e){
					continue;
				}

			}	
		
		}
		
		return blocks;
	}
	
	
	
	static Matrix addtionMatrix(Matrix matrix1,Matrix matrix2){
		int row = matrix1.Row;
		int col = matrix1.Col;
		int[][] addtion = new int[row][col];
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				addtion[i][j] = matrix1.blocks[i][j] + matrix2.blocks[i][j];
			}
		}
		Matrix addtionMatrix = new Matrix(row,col,addtion);
		return addtionMatrix;
	}
	
	static boolean allZero(int[] judge) {
		for(int i = 0; i < judge.length; i++) {
			if(judge[i] != 0) {
				return false;
			}
		}
		return true;
	}
	
	static void toString(Matrix matrix) {
		System.out.println(matrix.Row + ", " + matrix.Col);
		for(int i = 0; i < matrix.Row; i++) {
			System.out.print(i + 1 + " ");
			int[] judge = matrix.blocks[i];
			boolean allZero = Matrix.allZero(judge);
			if(allZero == true) {
				System.out.println(":");
				continue;
			}
			for(int j = 0; j < matrix.Col; j++) {
				if(matrix.blocks[i][j] != 0) {
					System.out.print(j + 1 + ":" + matrix.blocks[i][j] + " ");
				}
			}
			System.out.print("\n");
		}
	}
}


public class AddtionOfMatrices {
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		// get the Matrix1
		Scanner scanner = new Scanner(System.in);
		String a = scanner.nextLine();
		int row = Integer.parseInt(a.substring(0,a.indexOf(",")));
		int col = Integer.parseInt(a.substring(a.indexOf(" ")+1));

		String[] data1 = new String[row];
		String[] data2 = new String[row];
		for(int i = 0; i < row; i++) {
			data1[i] = scanner.nextLine();

		}
		String spaceline = scanner.nextLine();
		for(int i = 0; i < row; i++) {
			data2[i] = scanner.nextLine();

		}
		
		int[][] blocks1 = Matrix.generate_blocks(row, col, data1);
		Matrix matrix1 = new Matrix(row,col,blocks1);
		
		
		int[][] blocks2 = Matrix.generate_blocks(row, col, data2);
		Matrix matrix2 = new Matrix(row,col,blocks2);
		
	
		
		Matrix addtionMatrix = Matrix.addtionMatrix(matrix1, matrix2);
		
		
		
		Matrix.toString(addtionMatrix);
		
	}
}
