package com.inholland.frog.Recommendation;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;


import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class BeerRecommendation {

    public static final int ROW_LIMIT = 200000;

    public static final String outputPath = "preprocessed_beer_ratings.csv";

    public static void main(String[] args) throws Exception {

//        createProcessedCSV();

        Configuration conf = new Configuration();
        conf.set("dfs.data.dir", "./");
        conf.set("data.input.path", outputPath);
        conf.set("data.column.format", "UIR");
        conf.set("data.model.format", "csv");
        conf.set("data.model.splitter", "ratio");
        conf.set("data.splitter.trainset.ratio", "0.8");
        conf.set("rec.neighbors.knn.number", "5");

        // Build data model
        DataModel dataModel = new TextDataModel(conf);
        dataModel.buildDataModel();

        // Set similarity
        RecommenderSimilarity similarity = new PCCSimilarity();
        similarity.buildSimilarityMatrix(dataModel);

        // Set recommender context
        RecommenderContext context = new RecommenderContext(conf, dataModel);
        context.setSimilarity(similarity);

        // Build recommender model
        Recommender recommender = new UserKNNRecommender();
        recommender.setContext(context);
        recommender.recommend(context);


        Scanner scanner = new Scanner(System.in);


        while (true) {
            // Ask the user for a beer name
            System.out.println("Please enter a beer name (or type 'exit' to quit):");

            // Wait for user input
            String beerName = scanner.nextLine();

            // If the user types "exit", break out of the loop
            if (beerName.equalsIgnoreCase("exit")) {
                break;
            }

            // Get the Top-N beer recommendations based on the given beer name
            int topN = 3;
            Map<String, Double> topNBeerRecommendations = getTopNBeerRecommendations(beerName, dataModel, similarity, topN);

            // Print the Top-N beer recommendations
            System.out.println("Top " + topN + " Beer Recommendations for '" + beerName + "':");
            for (Map.Entry<String, Double> entry : topNBeerRecommendations.entrySet()) {
                System.out.println("Beer: " + entry.getKey() + ", Similarity: " + entry.getValue());
            }

            // Print an empty line for better readability
            System.out.println();
        }

        scanner.close();
        System.exit(1);
    }

    public static Map<String, Double> getTopNBeerRecommendations(String targetBeerName, DataModel dataModel, RecommenderSimilarity similarity, int topN) {
        // Find the item index for the target beer name
        int targetBeerIndex = dataModel.getItemMappingData().get(targetBeerName);

        // Get the similarity vector for the target beer
        SparseVector similarityVector = similarity.getSimilarityMatrix().row(targetBeerIndex);

        // Convert the similarity vector to a map of beer names and similarity scores
        Map<String, Double> similarityMap = new HashMap<>();
        for (VectorEntry entry : similarityVector) {
            int beerIndex = entry.index();
            double similarityScore = entry.get();
            String beerName = dataModel.getItemMappingData().inverse().get(beerIndex);
            similarityMap.put(beerName, similarityScore);
        }

        // Sort the map by similarity scores in descending order and get the Top-N similar beers
        Map<String, Double> sortedMap = similarityMap.entrySet().stream()
                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                .limit(topN)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        return sortedMap;
    }

    public static void createProcessedCSV() throws IOException {
        String inputPath = "beer_reviews.csv";
        preprocessCSV(inputPath, outputPath);
    }

    public static void preprocessCSV(String inputPath, String outputPath) throws IOException {
        try (FileReader fileReader = new FileReader(inputPath);
             FileWriter writer = new FileWriter(outputPath)) {
            int recordLimit = 0;
            CSVParser parser = CSVFormat.DEFAULT.withHeader().parse(fileReader);
            for (CSVRecord record : parser) {
                recordLimit++;
                if (recordLimit == ROW_LIMIT) break;
                String user = record.get("review_profilename");
                String item = record.get("beer_name").replaceAll(" ", "_").replaceAll(",", "-");
                String rating = record.get("review_overall");
                writer.write(user + "," + item + "," + rating + "\n");
            }
        }
    }
}