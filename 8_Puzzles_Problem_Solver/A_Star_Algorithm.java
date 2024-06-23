import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Scanner;

class Node implements Comparable<Node>{
	
	Board board;
	int moves,manhattan;
	Node prev;
	
	Node(Board board, int moves, int manhattan, Node prev){
		this.board = board;
		this.moves = moves;
		this.manhattan = manhattan;
		this.prev = prev;
	}
	
	int priority() {
		return manhattan+moves;
	}
	
	int moves() {
		return moves;
	}
	
	int manhattan() {
		return manhattan;
	}
	
	Node prev() {
		return prev;
	}	
	
	@Override
	public int compareTo(Node target) {
		return Integer.compare(this.priority(),target.priority());
	}

}




class Board{
	
	int[][] blocks;
	int Row,Col;
	
	Board(int[][] blocks) {
		
        this.blocks = blocks;

        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (blocks[row][col] == 0) {
                    this.Row = row;
                    this.Col = col;
                    return;
                }
            }
        }    
	}
	
      
	public int manhattan() {
        int result = 0;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (row == Row && col == Col) {
                    continue;
                }
                result += manhattan(row, col);
            }
        }
        return result;
    }

    private int manhattan(int row, int col) {
        int destVal = blocks[row][col] - 1;
        int destRow = destVal / 3;
        int destCol = destVal % 3;
        return Math.abs(destRow - row) + Math.abs(destCol - col);
    }
	
}


public class A_Star_Algorithm {
	
	static int[][] copyOf(int[][] matrix) {
        int[][] clone = new int[3][3];
        for (int row = 0; row < 3; row++) {
            clone[row] = matrix[row].clone();
        }
        return clone;
    }

	static void swap(int[][] a,int rowA, int colA, int rowB, int colB) {
		int swap = a[rowA][colA];
		a[rowA][colA] = a[rowB][colB];
		a[rowB][colB] = swap;
	}

	static LinkedList<Node> Neighbors(Node node) {
		
		LinkedList<Node> neighbors = new LinkedList<Node>();
		int Row = node.board.Row;
		int Col = node.board.Col;
		int moves = node.moves;
		int[][] north = copyOf(node.board.blocks);
		int[][] south = copyOf(node.board.blocks);
		int[][] west = copyOf(node.board.blocks);
		int[][] east = copyOf(node.board.blocks);
		if (Row > 0) {
            swap(north, Row, Col, Row - 1, Col);
            Board north1 = new Board(north);
            Node new_node = new Node(north1,moves+1,north1.manhattan(),node);
            neighbors.add(new_node);
        }
		
		if (Row < 2) {
            swap(south, Row, Col, Row + 1, Col);
            Board south1 = new Board(south);
            Node new_node = new Node(south1,moves+1,south1.manhattan(),node);
            neighbors.add(new_node);
        }
		
        if (Col > 0) {
            swap(west, Row, Col, Row, Col - 1);
            Board west1 = new Board(west);
            Node new_node = new Node(west1,moves+1,west1.manhattan(),node);
            neighbors.add(new_node);
        }
        
        if (Col < 2) {
            swap(east, Row, Col, Row, Col + 1);
            Board east1 = new Board(east);
            Node new_node = new Node(east1,moves+1,east1.manhattan(),node);
            neighbors.add(new_node);
        }  
		
        return neighbors;
	}
	
 
	
	static void toString(Node node) {
		
		int[][] blocks = node.board.blocks;
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				System.out.print(blocks[i][j]+" ");
			}
			System.out.println("\n");
		}
	}

	static LinkedList<Node> sol = new LinkedList<Node>();
	
	
	static Node sol(Node finalnode){
		if(finalnode == null) {
			return null;
		}
		sol.add(finalnode);
		return sol(finalnode.prev);	
	}

	
	public static void main(String[] args) {
		
		

		
		
		Scanner scanner = new Scanner(System.in);
		int[][] blocks = new int[3][3];
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				blocks[i][j] = scanner.nextInt();
			}
		}
		
		
		Board initial_board = new Board(blocks);
		Node initial_node = new Node(initial_board,0,initial_board.manhattan(),null);
		LinkedList<Node> a = new LinkedList<Node>();
		PriorityQueue<Node> pq = new PriorityQueue<Node>();

		pq.add(initial_node);
		
		while(true) {
			Node node = pq.poll();
		
			if(node== null||node.board.manhattan() == 0) {
				a.add(node);
				break;
			}
			LinkedList<Node> neighbors = Neighbors(node);
			
			for(int i = 0; i < neighbors.size(); i++) {
				pq.add(neighbors.get(i));		
				
			}
			
		}
		
		sol(a.get(0));
		
		for(int i = 0; i < sol.size(); i++) {
			int x = sol.size()- i-1;
			toString(sol.get(x));
			System.out.println("\n");
		}
		
	}
	
}	
