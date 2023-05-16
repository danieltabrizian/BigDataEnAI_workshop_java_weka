package com.inholland.frog.Clustering;

import com.inholland.frog.Clustering.HierarchicalClustering.Hierarchical;
import com.inholland.frog.Clustering.KMeans.KMeans;
import com.inholland.frog.Clustering.Regressions.LinearRegression;
import com.inholland.frog.Clustering.Regressions.SMOreg;

public class Clustering {
    public static void main(String[] args) {
        try{
            // Hierarchical.performDemo();
            KMeans.performDemo();
            // LinearRegression.performDemo();
            // SMOreg.performDemo();
        }catch (Exception e){
            System.out.print(e.getMessage());
        }
    }
}
