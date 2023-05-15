package com.inholland.frog.Classification;

import com.inholland.frog.Classification.NaiveBayes.NaiveBayesDemo;

public class Classification {
    public static void main(String[] args) {
        try{
            NaiveBayesDemo.process();
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }
}
