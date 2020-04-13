package Classification.knn;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.lang3.ArrayUtils;

public class Classifier_Knn {
	private double[][] trainingData;
	private int[] trainingLabels;
	private int k;
	enum distance {Euclidean, Cosine};
	public distance measure;
	
	public Classifier_Knn(int k, distance measure) {
		this.k = k;
		this.measure = measure;
	}
    
    public double getEuclideanDis(double[] a1, double[] a2)
    {
  	 double ans=0.0;
  	 for(int i=0;i<a1.length;i++)
  	 {
  		 double diff = a1[i]-a2[i];
  		 ans = ans + diff * diff;
  	 }
  	 
  	 return Math.sqrt(ans);
    }

	public double getCosineDis(double[] a1, double[] a2)
	{
		double dotProduct = 0.0;
		double a1Denom = 0.0;
		double a2Denom = 0.0;
		for(int i=0;i<a1.length;i++)
		{
			dotProduct = dotProduct + a1[i]*a2[i];
			a1Denom = a1Denom + a1[i]*a1[i];
			a2Denom = a2Denom + a2[i]*a2[i];
		}
		double sqrta1 = Math.sqrt(a1Denom);
		double sqrta2 = Math.sqrt(a2Denom);
		
		return Math.acos(dotProduct / (sqrta1*sqrta2));
	}
	
	public void fit_train(double[][] trainingData, int[] labels) {
		this.trainingData = trainingData;
		this.trainingLabels = labels;
	}
	
	public int[] fit(double[][] testData) {
		int testSize = testData.length;
		int[] labels = new int[testSize];
		for(int i=0; i<testSize; i++) {
			labels[i] = getLabel(testData[i]);
		}
		return labels;
	}

	private int getLabel(double[] testData) {
		int[] allLabels = new int[k];
		int categorisedLabel;
		double[][] trainedData = this.trainingData;
		int[] trainedLabel = this.trainingLabels;
		double currentDistance;
		int minIndex;
		for(int i=0; i<k; i++) {
			double minDistance = Integer.MAX_VALUE;
			minIndex = 0;
			for(int j=0; j<trainedData.length; j++) {
				if(this.measure == distance.Euclidean) {
					currentDistance = getEuclideanDis(testData, trainedData[j]);
				}
				else {
					currentDistance = getCosineDis(testData, trainedData[j]);
				}
				if(currentDistance < minDistance) {
					minDistance = currentDistance;
					minIndex = j;
				}
			}
			allLabels[i] = trainedLabel[minIndex];
			trainedLabel = ArrayUtils.remove(trainedLabel, minIndex);
			trainedData = ArrayUtils.remove(trainedData, minIndex);
		}
		categorisedLabel = getModeLabel(allLabels);
		return categorisedLabel;
	}
	
	private int getModeLabel(int[] labels) {
		int mode=0;
		HashMap<Integer,Integer> freqency = new HashMap<>();
		for(int i=0; i<k; i++) {
			if(freqency.containsKey(labels[i])) {
				freqency.put(labels[i], freqency.get(labels[i])+1);
			}
			else {
				freqency.put(labels[i], 1);
			}
		}
		int max=0;
		for(int key : freqency.keySet()) {
			int value = freqency.get(key);
			if(value > max) {
				max = value;
				mode = key;
			}
		}
		return mode;
	}
	
	public double[][] fit_fuzzy(double[][] testData){
		int testSize = testData.length;
		ArrayList<Integer> uniqueLabels = getUniqueLabels();
		double[][] categoryPercentages = new double[testSize][uniqueLabels.size()];
		for(int i=0; i<testSize; i++) {
			categoryPercentages[i] = getFuzzyLabels(testData[i], uniqueLabels);
		}
		return categoryPercentages;
	}

	private double[] getFuzzyLabels(double[] testData, ArrayList<Integer> uniqueLabels) {
		int uniqueLabelsSize = uniqueLabels.size();
		double[] fuzzyLabels = new double[uniqueLabelsSize];
		int[] allLabels = new int[k];
		int categorisedLabel;
		double[][] trainedData = this.trainingData;
		int[] trainedLabel = this.trainingLabels;
		double currentDistance;
		int minIndex;
		for(int i=0; i<k; i++) {
			double minDistance = Integer.MAX_VALUE;
			minIndex = 0;
			for(int j=0; j<trainedData.length; j++) {
				if(this.measure == distance.Euclidean) {
					currentDistance = getEuclideanDis(testData, trainedData[j]);
				}
				else {
					currentDistance = getCosineDis(testData, trainedData[j]);
				}
				if(currentDistance < minDistance) {
					minDistance = currentDistance;
					minIndex = j;
				}
			}
			allLabels[i] = trainedLabel[minIndex];
			trainedLabel = ArrayUtils.remove(trainedLabel, minIndex);
			trainedData = ArrayUtils.remove(trainedData, minIndex);
		}
		for(int label: allLabels) {
			fuzzyLabels[label] += 1;
		}
		for(int i=0; i<uniqueLabelsSize; i++) {
			fuzzyLabels[i] = Math.round((fuzzyLabels[i]*100)/k);
		}
		return fuzzyLabels;
	}

	private ArrayList<Integer> getUniqueLabels() {
		ArrayList<Integer> uniqueLabels =  new ArrayList<Integer>();
		for(int i=0; i<trainingLabels.length; i++) {
			if(!uniqueLabels.contains(trainingLabels[i])) {
				uniqueLabels.add(trainingLabels[i]);
			}
		}
		return uniqueLabels;
	}
}
