package com.inholland.frog.Clustering;

import com.inholland.frog.Clustering.HierarchicalClustering.Hierarchical;
import com.inholland.frog.Clustering.KMeans.KMeans;
import com.inholland.frog.Clustering.Regressions.LinearRegressionDemo;

public class Clustering {
    public static void main(String[] args) {
        try{
            // Hierarchical.performDemo();
            // KMeans.performDemo();
            // LinearRegressionDemo.performDemo();
            // SMOreg.performDemo();
        }catch (Exception e){
            System.out.print(e.getMessage());
        }
    }
}