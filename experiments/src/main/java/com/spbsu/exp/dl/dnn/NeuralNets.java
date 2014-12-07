package com.spbsu.exp.dl.dnn;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.DataUtils;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.dl.dnn.rectifiers.Rectifier;

import java.util.Random;

import static com.spbsu.exp.cuda.JcublasHelper.*;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:47
 */
public class NeuralNets {

  public FMatrix[] weights;
  public FMatrix[] activations;
  public FVector[] average;
  public Rectifier rectifier;
  public Rectifier outputRectifier;

  public NeuralNets(
      final @NotNull int[] layersDimensions,
      final int batchSize,
      final @NotNull Rectifier rectifier,
      final @NotNull Rectifier outputRectifier,
      final @NotNull Init initMethod
  ) {
    final int length = layersDimensions.length;

    weights = new FMatrix[length - 1];
    activations = new FMatrix[length];

    for (int i = 0; i < length - 1; i++) {
      weights[i] = new FArrayMatrix(layersDimensions[i + 1], layersDimensions[i] + 1);
      activations[i] = new FArrayMatrix(layersDimensions[i], batchSize);
    }
    activations[length - 1] = new FArrayMatrix(layersDimensions[length - 1], batchSize);

    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
    init(initMethod);
  }

  public NeuralNets(
      final @NotNull FMatrix[] weights,
      final @NotNull FMatrix[] activations,
      final @NotNull Rectifier rectifier,
      final @NotNull Rectifier outputRectifier
  ) {
    this.weights = weights;
    this.activations = activations;
    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
  }

  public NeuralNets(final @NotNull String path2model) {

  }

  private void init(final @NotNull Init initMethod) {
    switch (initMethod) {
      case RANDOM_SMALL : {
        final Random random = new Random();
        final FMatrix[] W = weights;

        for (int i = 0; i < W.length; i++) {
          final FMatrix wights = W[i];

          final int rows = wights.getRows();
          final int columns = wights.getColumns();
          final float scale = 8.f * (float) Math.sqrt(6. / (rows + columns - 1.));
          for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
              wights.set(j, k, (random.nextFloat() - 0.5f) * scale);
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

  public FMatrix batchForward(final @NotNull FMatrix X) {
    final FVector once = DataUtils.once(X.getColumns());
    activations[0] = DataUtils.extendAsBottomRow(X, once);

    final int last = weights.length;
    for (int i = 1; i < last; i++) {
      activations[i] = rectifier.activate(fMult(weights[i - 1], activations[i - 1]));

      //todo(ksenon): dropout, sparsity

      activations[i] = DataUtils.extendAsBottomRow(activations[i], once);
    }
    activations[last] = outputRectifier.activate(fMult(weights[last - 1], activations[last - 1]));

    return activations[last];
  }

  public void read(final @NotNull String path) {

  }

  public void write(final @NotNull String path) {

  }


}
