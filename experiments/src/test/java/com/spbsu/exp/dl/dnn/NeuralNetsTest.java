package com.spbsu.exp.dl.dnn;

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

/**
 * jmll
 * ksen
 * 07.December.2014 at 23:04
 */
public class NeuralNetsTest extends Assert {

  @Test
  public void testStartup() throws Exception {
    final FMatrix X = MNIST.getTrainDigits(1000);
    normalize(X);
    final FMatrix Y = MNIST.getTrainLabels(1000);
//    normalize(Y);

    final NeuralNets nn = new NeuralNets(new int[]{784, 1000, 10}, 1000, new SigmoidFA(), new IdenticalFA(), Init.RANDOM_SMALL);
    final NeuralNetsLearning nnl = new NeuralNetsLearning(nn, 2f, 20);

    nnl.batchLearn(X, Y);

    int error = 0;
    for (int i = 0; i < 1000; i++) {
      final FVector x = X.getColumn(i);
      final FVector y = Y.getColumn(i);
      final FVector a = nn.forward(x);

      if (getMaxIndex(y) != getMaxIndex(a)) {
        error++;
      }
    }
    System.out.println(error / 1000f);
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

}
