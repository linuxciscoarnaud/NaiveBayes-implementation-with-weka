/**
 * 
 */
package com.naivebayes0;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 * @author Arnaud
 *
 */
public class Classifier {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		// The filter
		StringToWordVector filter = new StringToWordVector();
		
		// Load dataset
		DataSource train_source = new DataSource("D:\\Development\\ML\\NaiveBayes0\\train_data.arff");
		DataSource test_source = new DataSource("D:\\Development\\ML\\NaiveBayes0\\test_data.arff");
		Instances train_dataset = train_source.getDataSet();
		Instances test_dataset = test_source.getDataSet();
		
		System.out.println(train_dataset.toString());
		System.out.println();
		
		// Set the datasets to the last attributes
		train_dataset.setClassIndex(train_dataset.numAttributes() - 1);
		test_dataset.setClassIndex(test_dataset.numAttributes() - 1);
		
		// Pass the datasets to the filter
		filter.setInputFormat(train_dataset);
		filter.setInputFormat(test_dataset);
		
		// Apply the filter
		train_dataset = Filter.useFilter(train_dataset, filter);
		Instances test_dataset_eval = Filter.useFilter(test_dataset, filter);
		
		System.out.println(train_dataset.toString());
		System.out.println();
		
		// Create and build the classifier
		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(train_dataset);
		
		// Evaluating the classifier
		for(int i = 0; i < test_dataset_eval.numInstances(); i++) {
			System.out.println(test_dataset.instance(i));
			double index = naiveBayes.classifyInstance(test_dataset_eval.instance(i));
			String className = train_dataset.classAttribute().value((int)index);
			System.out.println(className);
		}
	}
}
