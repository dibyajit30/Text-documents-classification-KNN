package Classification.knn;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;

public class documentMatrix {
    
	ArrayList<String> ordered_uniqueWords = new ArrayList<>();
	int no_of_docs;
	double[] idfs;
	
	public HashMap<String, Integer> getFreq(ArrayList<String> list, ArrayList<String> uniqueWords, HashMap<String, Integer> freq, boolean isTraining)
	{
		if(isTraining) {
			for(String s:list)
			   {
				   if(!uniqueWords.contains(s))
				   {
					   uniqueWords.add(s);
				   }
			   }
		}
		
		
		HashMap<String, Integer> freqCount = new HashMap<>();
		for(String s:list)
		{
			if(freqCount.containsKey(s))
			{
				freqCount.put(s, freqCount.get(s)+1);
			}
			else
			{
				freqCount.put(s,1);
			}
		}
		
		return freqCount;
	}
	
	
	public int[][] makeMatrix(ArrayList<HashMap<String, Integer>> listmap, ArrayList<String> set)
	{
	   int[][] documentMatrix = new int[listmap.size()][set.size()];
	   
		for(int i=0;i<listmap.size();i++)
	   {
		   int j=0;
			HashMap<String, Integer> map = listmap.get(i);
			for(String check:set)
		   {
				ordered_uniqueWords.add(check);
				if(!map.containsKey(check))
			   {
				   documentMatrix[i][j]=0; 
			   }
			   else
			   {
				   documentMatrix[i][j]=map.get(check); 
			   }
				j++;
		   }
	   }
		return documentMatrix;
	}
	
	public void setIDFS(int[][] documentMatrix, int length) {
		no_of_docs = length;
		idfs = new double[documentMatrix[0].length];
		int i=0;
		int j=0;
		int count;
		for(j=0;j<documentMatrix[0].length;j++)
		{
			count=0;
			for(i=0;i<length;i++)
			{
				if(documentMatrix[i][j]!=0)
				{
					count++;
				}
			}
			idfs[j]=count;
		}
	}
	
	public double[][] makeTM(int[][] documentMatrix)
	{
		double tf=0.0;
		double idf=0.0;
		int sum;
		int i=0;
		int j=0;
		
	    double[][] transformedMatrix = new double[documentMatrix.length][documentMatrix[0].length];
		
		
		
		for(i=0;i<documentMatrix.length;i++)
		{
			sum=0;
			for(j=0;j<documentMatrix[0].length;j++)
			{
			   sum = sum + documentMatrix[i][j]; 
			}
			for(j=0;j<documentMatrix[0].length;j++)
			{
				 tf = (double)documentMatrix[i][j]/sum;
				 idf = Math.log(no_of_docs/idfs[j]);
				 transformedMatrix[i][j] = tf*idf;
			}
		}
		
		return transformedMatrix;
	}
	
	
	public ArrayList<String> generateKeyWords(double[][] transformedMatrix)
	{
		double[][] intermediate_tm = new double[3][transformedMatrix[0].length];
		for(int j=0;j<transformedMatrix[0].length;j++)
		{	
			double sum=0.0;
			for(int i=0;i<8;i++)
			{
				sum = sum + transformedMatrix[i][j];
			}
			
			intermediate_tm[0][j] = sum;
			sum = 0.0;
			
			for(int i=8;i<16;i++)
			{
				sum = sum + transformedMatrix[i][j];
			}
			
			intermediate_tm[1][j] = sum;
			sum = 0.0;
			
			for(int i=16;i<24;i++)
			{
				sum = sum + transformedMatrix[i][j];
			}
			
			intermediate_tm[2][j] = sum;
		}
		
		ArrayList<String> keyWords = new ArrayList<>();
		String keyWord = "";
		for(double[] array:intermediate_tm)
		{
			double maximum = Double.MIN_VALUE;
			for(int i=0;i<array.length;i++)
			{
				if(array[i]>maximum)
				{
					maximum = array[i];
					keyWord = ordered_uniqueWords.get(i);
				}
			}
			keyWords.add(keyWord);
		}
		
		return keyWords;
	}	
}
