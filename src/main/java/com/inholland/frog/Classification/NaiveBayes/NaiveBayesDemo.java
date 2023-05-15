package com.inholland.frog.Classification.NaiveBayes;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class NaiveBayesDemo {
    /**
     * file names are defined
     */
    public static final String TRAINING_DATA_SET_FILENAME = "weather.nominal.arff";


    /**
     * This method is to load the data set.
     *
     * @param fileName
     * @return
     * @throws Exception
     */
    public static Instances getDataSet(String fileName) throws Exception {
        /**
         * we can set the file i.e., loader.setFile("finename") to load the data
         */

        StringToWordVector filter = new StringToWordVector();
        int classIdx = 1;

        /** the arffloader to load the arff file */
        ArffLoader loader = new ArffLoader();
        /** load the traing data */
        System.out.println(NaiveBayesDemo.class.getResourceAsStream("/"+fileName));

        loader.setSource(NaiveBayesDemo.class.getResourceAsStream("/"+fileName));

        Instances dataSet = loader.getDataSet();

        /** set the index based on the data given in the arff files */
        dataSet.setClassIndex(classIdx);
        filter.setInputFormat(dataSet);
        dataSet = Filter.useFilter(dataSet, filter);
        return dataSet;
    }

    /**
     * This method is used to process the input and return the statistics.
     *
     * @throws Exception
     */
    public static void process() throws Exception {

        Instances trainingDataSet = getDataSet(TRAINING_DATA_SET_FILENAME);
        Classifier classifier = new NaiveBayes();

        // Randomize the dataset //
        trainingDataSet.randomize(new java.util.Random(0));

        // Divide dataset into training and test data //
        int trainingDataSize = (int) Math.round(trainingDataSet.numInstances() * 0.66);
        int testDataSize = (int) trainingDataSet.numInstances() - trainingDataSize;

        // Create training data //
        Instances trainingInstances = new Instances(trainingDataSet,0,trainingDataSize);
        // Create test data //
        Instances testInstances = new Instances(trainingDataSet,trainingDataSize,testDataSize);

        // Set Target class. the attribute to predict is play//
        trainingInstances.setClassIndex(trainingInstances.numAttributes()-1);
        testInstances.setClassIndex(testInstances.numAttributes()-1);

        // Build Classifier //
        classifier.buildClassifier(trainingInstances);

        // Evaluation //
        Evaluation evaluation = new Evaluation(trainingInstances);
        evaluation.evaluateModel(classifier,testInstances);
        System.out.println(evaluation.toSummaryString("\nResults",false));
    }
}

