package com.spbsu.exp.dl.dnn;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.dl.Init;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:52
 */
public class NeuralNetsLearning {

  private NeuralNets nn;
  private float alpha;
  private float momentum;
  private float scalingAlpha;
  private float weightsPenalty;
  private float dropoutLevel;
  private int epochsNumber;

  public NeuralNetsLearning(
      final @NotNull NeuralNets nn,
      final float alpha,
      final float momentum,
      final float scalingAlpha,
      final float weightsPenalty,
      final float dropoutLevel,
      final int epochsNumber
  ) {
    this.nn = nn;
    this.alpha = alpha;
    this.momentum = momentum;
    this.scalingAlpha = scalingAlpha;
    this.weightsPenalty = weightsPenalty;
    this.dropoutLevel = dropoutLevel;
    this.epochsNumber = epochsNumber;
  }

  public void batchLearn(final @NotNull FMatrix X, final @NotNull FVector y, final int batchSize) {
    final int batchesNumber = y.getDimension() / batchSize;

    for (int i = 0; i < epochsNumber; i++) {
      for (int j = 0; j < batchesNumber; j++) {

      }
    }
  }

  private void backPropagation() {

  }

  public void init(final @NotNull Init initMethod) {
    switch (initMethod) {
      case RANDOM_SMALL : {
        final Random random = new Random();
        final FMatrix[] W = nn.weights;
        for (int i = 0; i < W.length; i++) {
          final FMatrix wights = W[i];

          final int rows = wights.getRows();
          final int columns = wights.getColumns();
          for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
              wights.set(j, k, (random.nextFloat() - 0.5f) * 8 * (float)Math.sqrt(6. / (rows + columns - 1.)));
            }
          }
        }
        break;
      }
      case DO_NOTHING : {
        break;
      }
    }
  }

}
