package com.spbsu.exp.dl.dnn;

import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.dl.dnn.rectifiers.Rectifier;

import static com.spbsu.exp.cuda.JcublasHelper.*;
/**
 * jmll
 * ksen
 * 23.November.2014 at 15:47
 */
public class NeuralNets {

  public FMatrix[] weights;
  public FMatrix[] activations;
  public Rectifier rectifier;
  public Rectifier outputRectifier;

  public NeuralNets(final @NotNull String path2model) {

  }

  public NeuralNets(
      final @NotNull int[] layersDimensions,
      final @NotNull Rectifier rectifier,
      final @NotNull Rectifier outputRectifier
  ) {
    final int length = layersDimensions.length;

    weights = new FMatrix[length - 1];
    activations = new FMatrix[length - 1];

    for (int i = 0; i < length - 1; i++) {
      weights[i] = new FArrayMatrix(layersDimensions[i], layersDimensions[i + 1] + 1);
      activations[i] = new FArrayMatrix(layersDimensions[i], layersDimensions[i + 1] + 1);
    }

    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
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

  public FVector predict(final @NotNull FVector x) {
    throw new UnsupportedOperationException("TODO");
  }

  public FMatrix batchPredict(final @NotNull FMatrix X) {
    activations[0] = X; //todo(ksenon): concat with 1 vector (1 | X)

    final int last = weights.length - 1;
    for (int i = 1; i < last; i++) {
      activations[i] = rectifier.activate(fMult(activations[i - 1], false, weights[i - 1], true));

      //todo(ksenon): dropout, sparsity
    }

    return outputRectifier.activate(fMult(activations[last - 1], false, weights[last - 1], true));
  }


}
