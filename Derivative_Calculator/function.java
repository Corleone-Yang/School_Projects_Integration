import java.util.Arrays;
import java.util.LinkedList;
import java.util.Scanner;

public class function {
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		Scanner scanner = new Scanner(System.in);
		String polynomial1 = scanner.nextLine();

		// eliminate the space
		String polynomial2 = polynomial1.replace(" ","");
		LinkedList<String> list1 = new LinkedList<String>();
		LinkedList<String> list2 = new LinkedList<String>();
		String polynomial = polynomial2.replace("-", "+-"); 


		//separate each items
		for (String element: polynomial.split("\\+")) {
			list1.add(element);
		}

		
		
		//the derivatives of a constant is 0, so ignore it.
		for (int i = 0; i < list1.size(); i++) {
			String w1 = list1.get(i);
			boolean judge1 = w1.contains("x");
			if (!judge1) {
				list1.remove(i);
			}
		}

		
		
		for (int i = 0; i < list1.size(); i++) {
			String w2 = list1.get(i);
			boolean judge2 = w2.contains("^");
			if (!judge2) {
				list1.set(i, w2+"^1");
			}
		}

		
		
		for (int i = 0; i < list1.size(); i++) {
			String w3 = list1.get(i);
			boolean judge3 = w3.contains("*");
			if (!judge3) {
				list1.set(i, "1*"+w3);
			}
		}

		
		
		int len = list1.size();
		int A[][] = new int [2][len];
		int B[][] = new int [2][len];

		
		for (int i = 0; i <= len-1; i++) {
			String a = list1.get(i);
			String b = a.replace("*x^", ",");
			String[] parts = b.split(",");
			String part1 = parts[0];
			String part2 = parts[1];
			A[0][i] = Integer.parseInt(part1);
			A[1][i] = Integer.parseInt(part2);
		}

		
		for (int i = 0; i <= len-1; i++) {
			//coefficient
			B[0][i] = A[0][i]*A[1][i];
			//power
			B[1][i] = A[1][i]-1;
		}
		
		
		for (int i = 0; i <= len-1; i++) {
			if (B[1][i] == 0) {
				list2.add(String.valueOf(B[0][i]));	
			} else if (B[1][i] == 1) {
				list2.add(String.valueOf(B[0][i])+"*x");
			} else {
				list2.add(String.valueOf(B[0][i])+"*x^"+String.valueOf(B[1][i]));				
			}
		}
			

		
		for (int i = 1; i <= list2.size()-1; i++) {
			String w4 = list2.get(i);
			boolean judge4 = w4.contains("-");
			if (!judge4) {
				list2.set(i, "+"+w4);
			}
		}

		
		String outcomes1 = list2.toString();
		String outcomes2 = outcomes1.replace("[", "");
		String outcomes3 = outcomes2.replace("]", "");
		String outcomes4 = outcomes3.replace(",", "");
		String outcomes = outcomes4.replace(" ","");
		
		System.out.println(outcomes);
	}
}			
