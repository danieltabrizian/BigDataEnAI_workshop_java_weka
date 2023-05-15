package com.inholland.frog.Classification;

import com.inholland.frog.Classification.NaiveBayes.NaiveBayesDemo;
import com.inholland.frog.Classification.RandomForest.RandomForestDemo;

public class Classification {
    public static void main(String[] args) {
        try{
            System.out.println("NaiveBayes:");
            NaiveBayesDemo.process();
            System.out.println("Random Forest:");
            RandomForestDemo.process();
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }
}
