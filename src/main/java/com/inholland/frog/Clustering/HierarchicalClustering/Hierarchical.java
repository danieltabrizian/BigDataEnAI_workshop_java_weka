package com.inholland.frog.Clustering.HierarchicalClustering;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;
import java.io.IOException;
import java.io.InputStream;
public class Hierarchical {
    public static final String FILENAME = "/data.csv";
    public static void performDemo() {
        try {
            // Load CSV file
            CSVLoader loader = new CSVLoader();
            InputStream inputStream = Hierarchical.class.getResourceAsStream(FILENAME);
            loader.setSource(inputStream);
            Instances data = loader.getDataSet();
            inputStream.close();

            // Perform clustering with HierarchicalClusterer
            HierarchicalClusterer hierarchicalClusterer = new HierarchicalClusterer();

            // Set the number of clusters
            int numClusters = 2;
            hierarchicalClusterer.setNumClusters(numClusters);

            // Set the linkage type
            SelectedTag linkageType = new SelectedTag(HierarchicalClusterer.BayesNet, HierarchicalClusterer.TAGS_LINK_TYPE);
            hierarchicalClusterer.setLinkType(linkageType);

            // Build the model
            hierarchicalClusterer.buildClusterer(data);

            // Evaluate the model
            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(hierarchicalClusterer);
            eval.evaluateClusterer(data);

            // Print cluster assignments
            System.out.println("Cluster assignments:");
            for (int i = 0; i < data.numInstances(); i++) {
                int cluster = hierarchicalClusterer.clusterInstance(data.instance(i));

                if (i % 10 == 0 || i == 0)
                    System.out.println("Cluster number: " + cluster);

                System.out.println("Instance " + i + " assigned to cluster " + cluster);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
