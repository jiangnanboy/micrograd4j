import micrograd.engine.Value;
import micrograd.nn.MLP;
import org.apache.commons.lang3.tuple.Pair;
import utils.CollectionUtil;
import utils.PropertiesReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.IntStream;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author sy
 * @date 2022/5/26 21:07
 */
public class Demo {
    public static void main(String[] args) {
        String dataPath =  Demo.class.getClassLoader().getResource(PropertiesReader.get("test_data")).getPath().replaceFirst("/", "");
        Pair<double[][], int[]> pair = getDatasets(dataPath);
        MLP model = initModel();
        optimization(model, pair.getLeft(), pair.getRight(), 0, 100);
    }

    /**
     * get dataset
     * @param dataPath
     * @return
     */
    public static Pair<double[][], int[]> getDatasets(String dataPath) {
        double[][] feats = new double[100][2];
        int[] labels = new int[100];
        try(BufferedReader br = Files.newBufferedReader(Paths.get(dataPath), StandardCharsets.UTF_8)) {
            String line;
            int index = 0;
            while(null != (line=br.readLine())) {
                String[] tokens = line.trim().split(" ");
                feats[index][0] = Double.parseDouble(tokens[0]);
                feats[index][1] = Double.parseDouble(tokens[1]);
                labels[index] = Integer.parseInt(tokens[2]);
                index ++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("features length -> " + feats.length + "\n" + "labels length -> " + labels.length);

        return Pair.of(feats, labels);
    }

    /**
     * init model
     * @return
     */
    public static MLP initModel() {
        List<Integer> nouts = CollectionUtil.newArrayList();
        nouts.add(16);
        nouts.add(16);
        nouts.add(1);
        // 2-layer neural network
        MLP model = new MLP(2, nouts);
        System.out.println(model.toString());
        System.out.println("number of parameters " + model.parameters().size());
        return model;
    }

    public static Pair<Value, Double> loss(MLP model, double[][] feats, int[] labels) {
       return loss(model, feats, labels, 0);
    }

    /**
     * loss
     * @param model
     * @param feats
     * @param labels
     * @param batchSize
     * @return
     */
    public static Pair<Value, Double> loss(MLP model, double[][]feats, int[] labels, int batchSize) {
        int[] trainLables = null;
        double[][] trainFeats = null;
        if(batchSize != 0) {
            int[] indexInt = IntStream.range(0, labels.length).toArray();
            int[][] indexIntTwo = new int[1][];
            indexIntTwo[0] = indexInt;
            INDArray indexArray = Nd4j.create(indexIntTwo);
            Nd4j.shuffle(indexArray, 0);
            indexInt = indexArray.toIntVector();
            int[] batchIndex = new int[batchSize];
            double[][] batchFeats = new double[batchSize][2];
            for(int i=0; i<batchSize; i++) {
                batchIndex[i] = indexInt[i];
                batchFeats[i] = feats[i];
            }
            trainFeats = batchFeats;
            trainLables = batchIndex;
        } else {
            trainFeats = feats;
            trainLables = labels;
        }
        List<List<Value>> inputs = CollectionUtil.newArrayList();
        for(double[] feat : trainFeats) {
            List<Value> valueList = CollectionUtil.newArrayList();
            valueList.add(new Value(feat[0]));
            valueList.add(new Value(feat[1]));
            inputs.add(valueList);
        }
        // forward the model to get scores
        List<Value> scores = CollectionUtil.newArrayList();
        for(List<Value> values : inputs) {
            scores.add(model.apply(values).get(0));
        }
        List<Value> losses = CollectionUtil.newArrayList();
        Value sumValue = new Value(0);
        for(int i = 0; i<scores.size();i++) {
            int yi = trainLables[i];
            Value scorei = scores.get(i);
            Value v = (scorei.mul(yi*(-1)).add(1)).relu();
            losses.add(v);
            sumValue = sumValue.add(v);
        }

        Value dataLoss = sumValue.mul(1.0/losses.size());

        // L2 regularization
        double alpha = 1e-4;
        Value modelParams = new Value(0);
        for(Value mp : model.parameters()) {
            modelParams = modelParams.add(mp.mul(mp));
        }

        Value regLoss = modelParams.mul(alpha);
        Value totalLoss = dataLoss.add(regLoss);

        // also get accuracy
        List<Integer> accuracyList = CollectionUtil.newArrayList();
        for(int i = 0; i<scores.size();i++) {
            int yi = trainLables[i];
            Value scorei = scores.get(i);
            int yiLable = 0;
            int scoreiLabel = 0;
            if(yi > 0) {
                yiLable = 1;
            }
            if(scorei.data > 0) {
                scoreiLabel = 1;
            }
            if(yiLable == scoreiLabel) {
                accuracyList.add(1);
            } else {
                accuracyList.add(0);
            }

        }
        int accuracySum = accuracyList.stream().reduce(Integer::sum).get();
        return Pair.of(totalLoss, (double)accuracySum/accuracyList.size());
    }

    /**
     * optimization
     * @param model
     * @param feats
     * @param labels
     * @param batchSize
     * @param step
     */
    public static void optimization(MLP model, double[][]feats, int[] labels, int batchSize, int step) {
        for(int s=0; s<step; s++) {
            // forward
            Pair<Value, Double> pair = loss(model, feats, labels, batchSize);
            Value totalLoss = pair.getLeft();
            double acc = pair.getRight();
            // backward
            model.zeroGrad();
            totalLoss.backward();
            // update(sgd)
            double learningRate = 1.0 - 0.9*s/100;
            for(Value value : model.parameters()) {
                value.data -= learningRate * value.grad;
            }
            if(s % 1 == 0) {
                System.out.println("step " + s + " loss " + totalLoss.data + ", accuracy " + acc*100+"%");
            }
        }
    }

}
