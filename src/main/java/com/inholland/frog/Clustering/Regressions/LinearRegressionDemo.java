package com.inholland.frog.Clustering.Regressions;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class LinearRegressionDemo {
    public static final String FILENAME = "/stock.arff";
    public static void performDemo() {
        try {
            // Load the ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setSource(LinearRegressionDemo.class.getResourceAsStream(FILENAME));
            Instances dataset = loader.getDataSet();

            // Set the class index (assuming the last attribute is the target variable)
            dataset.setClassIndex(dataset.numAttributes() - 2);

            // Define the training and testing data splits
            int trainSize = (int) Math.round(dataset.numInstances() * 0.85);
            int testSize = dataset.numInstances() - trainSize;
            Instances trainDataset = new Instances(dataset, 0, trainSize);
            Instances testDataset = new Instances(dataset, trainSize, testSize);

            // Create and build the linear regression model
            LinearRegression model = new LinearRegression();
            model.buildClassifier(trainDataset);

            // Print the coefficients of the linear regression model
            double[] coefficients = model.coefficients();
            System.out.println("Linear Regression Coefficients:");
            for (int i = 0; i < coefficients.length - 1; i++) {
                System.out.println("Coefficient " + i + ": " + coefficients[i]);
            }

            // Print the intercept of the linear regression model
            System.out.println("Intercept: " + coefficients[coefficients.length - 1]);

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