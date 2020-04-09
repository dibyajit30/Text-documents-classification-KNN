package Classification.knn;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import Classification.knn.Classifier_Kmeans.distance;

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
        int[] test_y = new int[] {0,0,0,0,1,1,2,2,0,0};
        
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
   	   
   	   HashMap<String, Integer> map = dm.getFreq(list, uniqueWords, freq);
   	   listmap.add(map);
      }
   	 
   	for(File file: testFolder.listFiles())
  	 { 
  	   freq = new HashMap<>();
  	   ArrayList<String> ans = new ArrayList<>();
  	   list = pp.readFromFile(stopwords, file);
  	   
  	   HashMap<String, Integer> map = dm.getFreq(list, uniqueWords, freq);
  	   listmap.add(map);
     }
   	 
   	  
   	  int[][] documentMatrix = dm.makeMatrix(listmap, uniqueWords);
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
   	Classifier_Kmeans classifier = new Classifier_Kmeans(3, Classifier_Kmeans.distance.Cosine);
   	classifier.fit_train(train_x, train_y);
   	pred_y = classifier.fit(test_x);
   	for(int i=0; i<testSetSize; i++) {
   		System.out.print(pred_y[i] + " ");
   	}
   	
   	Performance performance = new Performance();
   	performance.accuracy(test_y, pred_y);
   	performance.precision(test_y, pred_y);
   	performance.recall(test_y, pred_y);

    }
}
