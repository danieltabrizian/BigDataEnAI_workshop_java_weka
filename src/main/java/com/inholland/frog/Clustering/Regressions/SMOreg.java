package com.inholland.frog.Clustering.Regressions;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class SMOreg {
    private static final String FILENAME = "/stock.arff";

    public static void performDemo() {
        try {
            // Load the ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setSource(SMOreg.class.getResourceAsStream(FILENAME));
            Instances dataset = loader.getDataSet();

            // Set the class index (assuming the last attribute is the target variable)
            dataset.setClassIndex(dataset.numAttributes() - 2);

            // Define the training and testing data splits
            int trainSize = (int) Math.round(dataset.numInstances() * 0.85);
            int testSize = dataset.numInstances() - trainSize;
            Instances trainDataset = new Instances(dataset, 0, trainSize);
            Instances testDataset = new Instances(dataset, trainSize, testSize);

            // Create and build the SMOreg regression model
            // Sequential Minimal Optimization for Regression
            weka.classifiers.functions.SMOreg model = new weka.classifiers.functions.SMOreg();
            model.buildClassifier(trainDataset);

            // Make predictions on the test instances
            for (int i = 0; i < testDataset.numInstances(); i++) {
                double prediction = model.classifyInstance(testDataset.instance(i));
                System.out.println("Prediction for instance " + i + ": " + prediction);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}