package Classification.knn;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Performance {
    public void accuracy(int[] true_labels, int[] pred_labels) {
    	int length = true_labels.length;
    	int correct = 0;
    	for(int i=0; i<length; i++) {
    		if(true_labels[i] == pred_labels[i]) {
    			correct += 1;
    		}
    	}
    	System.out.println("Accuracy: " + ((double)correct/length)*100 + "%");
    }
    
    public void precision(int[] true_labels, int[] pred_labels) {
    	ArrayList<Integer> uniqueLabels = new ArrayList<Integer>();
    	int length = true_labels.length;
    	for(int i=0; i<length; i++) {
    		if(! uniqueLabels.contains(true_labels[i])) {
    			uniqueLabels.add(true_labels[i]);
    		}
    	}
    	int truePositive=0;
    	int falsePositive=0;
    	double precision;
    	for(int category: uniqueLabels) {
    		for(int i=0; i<length; i++) {
    			if((true_labels[i] == category && (pred_labels[i] == category))) {
    				truePositive += 1;
    			}
    			else if((true_labels[i] != category && (pred_labels[i] == category))) {
    				falsePositive += 1;
    			}
    		}
    		precision = (double)truePositive/(truePositive + falsePositive);
    		System.out.println("Precision for category " + category + " = " + precision);
    	}
    }
    
    public void recall(int[] true_labels, int[] pred_labels) {
    	ArrayList<Integer> uniqueLabels = new ArrayList<Integer>();
    	int length = true_labels.length;
    	for(int i=0; i<length; i++) {
    		if(! uniqueLabels.contains(true_labels[i])) {
    			uniqueLabels.add(true_labels[i]);
    		}
    	}
    	int truePositive=0;
    	int falseNegative=0;
    	double recall;
    	for(int category: uniqueLabels) {
    		for(int i=0; i<length; i++) {
    			if((true_labels[i] == category && (pred_labels[i] == category))) {
    				truePositive += 1;
    			}
    			else if((true_labels[i] == category && (pred_labels[i] != category))) {
    				falseNegative += 1;
    			}
    		}
    		recall = (double)truePositive/(truePositive + falseNegative);
    		System.out.println("Recall for category " + category + " = " + recall);
    	}
    }
}
