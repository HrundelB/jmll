package com.spbsu.exp.dl.dbn;

import com.spbsu.exp.dl.Init;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

import static com.spbsu.exp.cuda.JcublasHelper.*;
import static com.spbsu.exp.cuda.JcudaVectorInscale.*;

/**
 * jmll
 * ksen
 * 20.November.2014 at 23:48
 */
public class DeepBeliefNetsLearning {

  private DeepBeliefNets deepBeliefNets;
  private float alpha;
  private int epochs;

  public DeepBeliefNetsLearning(final @NotNull DeepBeliefNets deepBeliefNets, final float alpha, final int epochs) {
    this.deepBeliefNets = deepBeliefNets;
    this.alpha = alpha;
    this.epochs = epochs;
  }

  public void learn(final @NotNull FMatrix X) {
    final int examplesNumber = X.getColumns();
    final FMatrix[] weights = deepBeliefNets.weights;
    final FMatrix[] cachedOutputs = new FMatrix[weights.length];

    for (int layer = 0; layer < weights.length; layer++) {
      if(layer == 0) {
        cachedOutputs[layer] = X;
      }
      else {
        final FMatrix W = weights[layer - 1];
        final FMatrix prev = cachedOutputs[layer - 1];
        final FMatrix next = new FArrayMatrix(weights[layer - 1].getColumns(), examplesNumber);

        for (int i = 0; i < examplesNumber; i++) {
          final FVector example = prev.getColumn(i);
          next.setColumn(i, FHeaviside(fMult(example, W)));
        }
        cachedOutputs[layer] = next;
      }

      FMatrix W = weights[layer];
      final FMatrix examples = cachedOutputs[layer];

      for (int epoch = 0; epoch < epochs; epoch++) {
        for (int example = 0; example < examplesNumber; example++) {
          final FVector v0 = examples.getColumn(example);
          final FVector h0 = FHeaviside(fMult(v0, W));

          final FVector v1 = FHeaviside(fMult(W, h0));
          final FVector h1 = FHeaviside(fMult(v1, W));

          // W = W + alpha * (v * trans(h) - v' * trans(h'))
          W = fSum(W, fScale(fSubtr(fMult(v0, h0), fMult(v1, h1)), alpha));
        }
        System.out.println("Epoch " + epoch + " on " + layer + " layer done.");
      }
      System.out.println("Layer " + layer + " trained.");
    }
  }

  public void init(final @NotNull Init init) {
    final FMatrix[] weights = deepBeliefNets.weights;

    switch (init) {
      case RANDOM_SMALL: {
        final Random random = new Random();
        final float scale = 0.01f;
        final float shift = scale / 2.f;

        for (int i = 0; i < weights.length; i++) {
          final FMatrix W = weights[i];

          for (int j = 0; j < W.getRows(); j++) {
            for (int k = 0; k < W.getColumns(); k++) {
              W.set(j, k, random.nextFloat() * scale - shift);
            }
          }
        }
        break;
      }
      case IDENTITY: {
        for (int i = 0; i < weights.length; i++) {
          final FMatrix W = weights[i];

          for (int j = 0; j < W.getRows(); j++) {
            for (int k = 0; k < W.getColumns(); k++) {
              W.set(j, k, 1.f);
            }
          }
        }
        break;
      }
      case DO_NOTHING: {
        break;
      }
    }
  }

}
