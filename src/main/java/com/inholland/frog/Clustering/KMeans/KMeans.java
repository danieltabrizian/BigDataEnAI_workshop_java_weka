package com.inholland.frog.Clustering.KMeans;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
public class KMeans {
    public static final String FILENAME = "/data.csv";
        public static void performDemo(){
        try {
            // Load CSV file
            CSVLoader loader = new CSVLoader();
            loader.setSource(KMeans.class.getResourceAsStream(FILENAME));
            Instances data = loader.getDataSet();

            // Perform clustering with SimpleKMeans
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setNumClusters(4); // Set the number of clusters

            // Build the model
            kMeans.buildClusterer(data);

            // Evaluate the model
            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(kMeans);
            eval.evaluateClusterer(data);

            // Print cluster assignments
            System.out.println("Cluster assignments:");
            for (int i = 0; i < data.numInstances(); i++) {
                int cluster = kMeans.clusterInstance(data.instance(i));

                if (i % 10 == 0 || i == 0)
                    System.out.println("Clusternumber: " + cluster);

                System.out.println("Instance " + i + " assigned to cluster " + cluster);
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}