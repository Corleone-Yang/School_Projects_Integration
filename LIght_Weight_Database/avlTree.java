import java.util.Scanner;
import java.util.Stack;
import java.io.*;


public class avlTree {
	Node root;
	class Node{
		
		int data,height;
		Node left,right;
		
		Node(int data){
			this.data = data;
		}
	}
	
	
	int height(Node node) {
		return node != null ? node.height : -1;
	}
	
	
	int max(int a,int b) {
		if(a >= b) {
			return a;
		}else {
			return b;
		}
	}
	
	void updateHeight(Node node) {
		int leftChildHeight = height(node.left);
		int rightChildHeight = height(node.right);
		node.height = max(leftChildHeight, rightChildHeight)+1;
	}
	
	
	int balanceFactor(Node node) {
		if(node == null) {
			return 0;
		}else {
			return height(node.right) - height(node.left);
		}	
	}
	
	
	Node rotateLeft(Node node) {
		Node rightChild = node.right;
		
		node.right = rightChild.left;
		rightChild.left = node;
		
		updateHeight(node);
		updateHeight(rightChild);
		
		return rightChild;
	}
	
	
	Node rotateRight(Node node) {
		Node leftChild = node.left;
		
		node.left = leftChild.right;
		leftChild.right = node;
		
		updateHeight(node);
		updateHeight(leftChild);
		
		return leftChild;
		
	}
	private Node rebalance(Node node) {
		  int balanceFactor = balanceFactor(node);

		  // Left-heavy?
		  if (balanceFactor < -1) {
		    if (balanceFactor(node.left) <= 0) {    // Case 1
		      // Rotate right
		      node = rotateRight(node);
		    } else {                                // Case 2
		      // Rotate left-right
		      node.left = rotateLeft(node.left);
		      node = rotateRight(node);
		    }
		  }

		  // Right-heavy?
		  if (balanceFactor > 1) {
		    if (balanceFactor(node.right) >= 0) {    // Case 3
		      // Rotate left
		      node = rotateLeft(node);
		    } else {                                 // Case 4
		      // Rotate right-left
		      node.right = rotateRight(node.right);
		      node = rotateLeft(node);
		    }
		  }

		  return node;
		}
	
	
	void insert(int data) {
		root = insert_Recursive(root,data);
	}
	Node insert_Recursive(Node root, int data) {
		
		if(root == null) {
			return new Node(data);
		}
		if(data <= root.data) {
			root.left = insert_Recursive(root.left,data);
		}else if(data > root.data) {
			root.right = insert_Recursive(root.right,data);
		}
		updateHeight(root);
		
		return rebalance(root);
	}
	
	
	void delete(int data) {
		root = delete_Recursive(root,data);
	}
	Node delete_Recursive(Node root, int data) {
		if(root == null) {
			return null;
		}
		if(data < root.data) {
			root.left = delete_Recursive(root.left,data);
		}else if(data > root.data) {
			root.right = delete_Recursive(root.right,data);
		}else {
			if(root.left == null) {
				return root.right;
			}else if(root.right == null) {
				return root.left;
			}
			
			root.data = minValue(root.right);
			root.right = delete_Recursive(root.right,root.data);
		}
		
		updateHeight(root);
		return rebalance(root);
	}	

	int minValue(Node root) {
		int minval = root.data;
		
		while(root.left !=null) {
			minval = root.left.data;
			root = root.left;
		}
		return minval;
	}
	
	void inOrder(Node node) {
		if(node != null) {
			inOrder(node.left);
			System.out.println(node.data);
			inOrder(node.right);
		}
	}
	
	void successor(int data) {
		int res = inOrderSuccessor(root,data);
		System.out.println(res);
	}
	int inOrderSuccessor(Node root,int p){
		Node res = null;
		while(root != null) {
			if(root.data <= p) {
				root = root.right;
			}else {
				res = root;
				root = root.left;
			}
		}
		return res.data;
	}
	
	
	void predecessor(int data) {
		int res = inOrderPredecessor(root,data);
		System.out.println(res);
	}
	int inOrderPredecessor(Node root,int p) {
		Node res = null;
		while(root !=null) {
			if(root.data >= p) {
				root = root.left;
			}else {
				res = root;
				root = root.right;
			}
		}
		return res.data;
	}
	
	
	void findRank(int x) {
		int res = rank(root,x)+1;
		System.out.println(res);
	}
	int rank(Node root,int x) {
		if(root == null) {
			return 0;
		}
		//right
		if(root.data < x) {
			return 1 + rank(root.left,x) + rank(root.right,x);
		}else {
			return rank(root.left,x);
		}
	}
	
	
	void findxthSmallest(int x) {
		int res = xthSmallest(root,x);
		System.out.println(res);
	}
	int xthSmallest(Node root,int x) {
		Stack<Node> stack = new Stack<>();
		
		while(true) {
			while(root != null) {
				stack.push(root);
				root = root.left;
			}
			
			root = stack.pop();
			if(--x == 0) {
				return root.data;
			}
			root = root.right;
		}
	}
	
	
	public static void main(String[] args) throws FileNotFoundException {
		avlTree a = new avlTree();
		
		//String fileName = "5(2).in";
		//Scanner scanner = new Scanner(new FileReader(fileName));
		Scanner scanner = new Scanner(System.in);
		int b = scanner.nextInt();
		for(int i = 0; i < b; i++) {
			int c = scanner.nextInt();
			int d = scanner.nextInt();
			if(c == 1) {
				a.insert(d);
			}else if(c == 2) {
				a.delete(d);
			}else if(c == 3) {
				a.findRank(d);
			}else if(c == 4) {
				a.findxthSmallest(d);
			}else if(c == 5) {
				a.predecessor(d);
			}else if(c == 6) {
				a.successor(d);
			}
		}
	}

}
