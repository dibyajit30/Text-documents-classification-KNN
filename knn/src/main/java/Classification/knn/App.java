package Classification.knn;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class App 
{
	public static void main( String[] args ) throws IOException
    {
    	// This puts all the stop words into hashset.	
    	ArrayList<String> uniqueWords = new ArrayList<>();
        HashMap<String, Integer> freq;
        File trainingFolder = new File("..\\\\knn\\\\resources\\\\data\\\\training");
        File testFolder = new File("..\\\\knn\\\\resources\\\\data\\\\test");
        int trainingSetSize = trainingFolder.listFiles().length;
        int testSetSize = testFolder.listFiles().length;
        
        //Training labels
        int[] train_y = new int[] {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2};
        //Test labels
        int[] test_y = new int[] {2,2,2,2,1,1,0,0,1,1};
        
        PreProcessing pp = new PreProcessing();
        documentMatrix dm = new documentMatrix();
        ArrayList<String> list = new ArrayList<>();
        
        HashSet<String> stopwords = pp.getStopWords();
       
        ArrayList<HashMap<String, Integer>> listmap = new ArrayList<>();
   	 for(File file: trainingFolder.listFiles())
   	 { 
   	   freq = new HashMap<>();
   	   ArrayList<String> ans = new ArrayList<>();
   	   list = pp.readFromFile(stopwords, file);
   	   
   	   HashMap<String, Integer> map = dm.getFreq(list, uniqueWords, freq, true);
   	   listmap.add(map);
      }
   	 
   	for(File file: testFolder.listFiles())
  	 { 
  	   freq = new HashMap<>();
  	   ArrayList<String> ans = new ArrayList<>();
  	   list = pp.readFromFile(stopwords, file);
  	   
  	   HashMap<String, Integer> map = dm.getFreq(list, uniqueWords, freq, false);
  	   listmap.add(map);
     }
   	 
   	  
   	  int[][] documentMatrix = dm.makeMatrix(listmap, uniqueWords);
   	  dm.setIDFS(documentMatrix, trainingSetSize);
   	  double[][] transformedMatrix = dm.makeTM(documentMatrix);
   	  
   	// Train test split
   	double[][] train_x = new double[trainingSetSize][];
   	double[][] test_x = new double[testSetSize][];
   	
   	for(int i=0; i<trainingSetSize; i++) {
   		train_x[i] = transformedMatrix[i];
   	}
   	
   	for(int i=trainingSetSize; i < trainingSetSize + testSetSize; i++) {
   		test_x[i - trainingSetSize] = transformedMatrix[i];
   	}
   	  
   	int[] pred_y = new int[testSetSize];
   	/*
   	 * Instantiating knn. Constructor takes two parameters - K value(greater than zero),
   	 * Euclidean/Cosine distance for document similarity.
   	 */
   	Classifier_Knn classifier = new Classifier_Knn(4, Classifier_Knn.distance.Cosine);
   	classifier.fit_train(train_x, train_y);
   	pred_y = classifier.fit(test_x);
   	System.out.println("Predicted labels");
   	for(int i=0; i<testSetSize; i++) {
   		System.out.print(pred_y[i] + " ");
   	}
   	System.out.println();
   	System.out.println("True labels");
   	for(int i=0; i<testSetSize; i++) {
   		System.out.print(test_y[i] + " ");
   	}
   	System.out.println();
   	Performance performance = new Performance();
   	performance.accuracy(test_y, pred_y);
   	performance.precision(test_y, pred_y);
   	performance.recall(test_y, pred_y);
    
	/*
	 * Predicting through fuzzy-knn.
	 */
	double[][] pred_fuzzy = classifier.fit_fuzzy(test_x);
	int uniqueLabelsSize = pred_fuzzy[0].length;
	System.out.println("Fuzzy labels");
	for(int i=0; i<testSetSize; i++) {
		System.out.print("Document:"+(i+1));
		for(int j=0; j<uniqueLabelsSize; j++) {
			System.out.print(" Category:" + j + " " + pred_fuzzy[i][j] + "%");
		}
		System.out.println();
	}
    }
}
