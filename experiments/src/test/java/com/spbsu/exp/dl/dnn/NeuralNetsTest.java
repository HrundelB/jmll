package com.spbsu.exp.dl.dnn;

import com.spbsu.commons.system.RuntimeUtils;
import com.spbsu.exp.cuda.process.functions.floats.TanhFA;
import org.junit.Test;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import com.spbsu.exp.cuda.process.functions.floats.IdenticalFA;
import com.spbsu.exp.cuda.process.functions.floats.SigmoidFA;
import com.spbsu.exp.dl.datasets.MNIST;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import org.junit.Assert;

import java.io.File;
import java.util.Arrays;

/**
 * jmll
 * ksen
 * 07.December.2014 at 23:04
 */
public class NeuralNetsTest extends Assert {

  @Test
  public void testStartup() throws Exception {
    final int examples = 60_000;
    final int testExamples = 10_000;
    final FMatrix X = MNIST.getTrainDigits(examples);
//    normalize(X);
    final FMatrix Y = MNIST.getTrainLabels(examples);
//    normalize(Y);

    final FMatrix testX = MNIST.getTestDigits(testExamples);
    final FMatrix testY = MNIST.getTestLabels(testExamples);

    final NeuralNets nn = new NeuralNets(new int[]{784, 500, 10}, 10, new SigmoidFA(), new IdenticalFA(), Init.RANDOM_SMALL);
    final NeuralNetsLearning nnl = new NeuralNetsLearning(nn, 0.01f, 1);

    for (int ep = 0; ep < 15; ep++) {
      nnl.batchLearn(X, Y);

      int maxError = 0;
      int minError = 0;
      int indexMax = 0;
      int indexMax1 = 0;
      int indexMin = 0;
      for (int i = 0; i < testExamples; i++) {
        final FVector x = testX.getColumn(i);
        final FVector y = testY.getColumn(i);
        final FVector a = nn.forward(x);

        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        float yMax = Float.MIN_VALUE;
        for (int j = 0; j < y.getDimension(); j++) {
          final float answer = a.get(j);
          final float target = y.get(j);
          if (answer < min) {
            min = answer;
            indexMin = j;
          }
          if (answer > max) {
            max = answer;
            indexMax = j;
          }
          if (target > yMax) {
            yMax = target;
            indexMax1 = j;
          }
        }
        if (indexMax != indexMax1) {
          maxError++;
        }
        if (indexMin != indexMax1) {
          minError++;
        }
      }
      System.out.println("Min-error: " + (minError / (float)testExamples) + ", Max-error: " + (maxError / (float)testExamples));
    }
    nn.write(new File("experiments/src/test/data/dl/dnn.data").getAbsolutePath());
  }

  private FMatrix normalize(final FMatrix A) {
    final int rows = A.getRows();
    final int columns = A.getColumns();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        A.set(i, j, A.get(i, j) / 255.f);
      }
    }
//    final FVector mean = new FArrayVector(rows);
//    for (int i = 0; i < rows; i++) {
//      for (int j = 0; j < columns; j++) {
//        mean.set(i, mean.get(i) + A.get(i, j));
//      }
//      mean.set(i, mean.get(i) / columns);
//    }
//    final FVector std = new FArrayVector(rows);
//    for (int i = 0; i < rows; i++) {
//      for (int j = 0; j < columns; j++) {
//        std.set(i, std.get(i) + (float)Math.pow(A.get(i, j) - mean.get(i), 2.));
//      }
//      final float sqrt = (float) Math.sqrt(std.get(i) / columns);
//      std.set(i, sqrt < 0.001f ? 0.001f : sqrt);
//    }
//    for (int i = 0; i < rows; i++) {
//      for (int j = 0; j < columns; j++) {
//        A.set(i, j, (A.get(i, j) - mean.get(i)) / std.get(i));
//      }
//    }

    return A;
  }

  private int getMaxIndex(final FVector a) {
    int index = 0;
    float max = Float.MIN_VALUE;
    for (int i = 0; i < a.getDimension(); i++) {
      if (max < a.get(i)) {
        index = i;
      }
    }
    return index;
  }

  @Test
  public void testName() throws Exception {
    final FArrayMatrix W0 = setMatrix(3, 4);
    final FArrayMatrix W1 = setMatrix(5, 3);
    final FArrayMatrix W2 = setMatrix(1, 5);
    final NeuralNets nn = new NeuralNets(
        new FMatrix[]{
            W0, W1, W2
        },
        new FMatrix[]{},
        new SigmoidFA(),
        new IdenticalFA()
    );

    final File model = new File(RuntimeUtils.getSysTmpDir(), "/tmp-dnn-model." + System.currentTimeMillis());
    nn.write(model.getAbsolutePath());

    final NeuralNets nn2 = new NeuralNets(model.getAbsolutePath());

    System.out.println();
  }

  private FArrayMatrix setMatrix(final int rows, final int columns) {
    final FArrayMatrix matrix = new FArrayMatrix(rows, columns);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        matrix.set(i, j, i * columns + j);
      }
    }
    return matrix;
  }

  @Test
  public void testName2() throws Exception {
    final int testExamples = 10000;
    final FMatrix testX = MNIST.getTestDigits(testExamples);
    final FMatrix testY = MNIST.getTestLabels(testExamples);

    final NeuralNets nn = new NeuralNets("experiments/src/test/data/dl/dnn.data");

    int maxError = 0;
    int minError = 0;
    int indexMax = 0;
    int indexMax1 = 0;
    int indexMin = 0;
    for (int i = 0; i < testExamples; i++) {
      final FVector x = testX.getColumn(i);
      final FVector y = testY.getColumn(i);
      final FVector a = nn.forward(x);

      float min = Float.MAX_VALUE;
      float max = Float.MIN_VALUE;
      float yMax = Float.MIN_VALUE;
      for (int j = 0; j < y.getDimension(); j++) {
        final float answer = a.get(j);
        final float target = y.get(j);
        if (answer < min) {
          min = answer;
          indexMin = j;
        }
        if (answer > max) {
          max = answer;
          indexMax = j;
        }
        if (target > yMax) {
          yMax = target;
          indexMax1 = j;
        }
      }
      if (indexMax != indexMax1) {
        maxError++;
      }
      if (indexMin != indexMax1) {
        minError++;
      }
    }
    System.out.println("Min-error: " + (minError / (float) testExamples) + ", Max-error: " + (maxError / (float) testExamples));
  }
}
