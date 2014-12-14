package com.spbsu.exp.dl.dnn;

import com.spbsu.exp.cuda.JcudaVectorInscale;
import com.spbsu.exp.cuda.data.DataUtils;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

import static com.spbsu.exp.cuda.JcublasHelper.*;
import static com.spbsu.exp.cuda.JcublasHelper.fScale;
import static com.spbsu.exp.cuda.JcublasHelper.fSubtr;
import static com.spbsu.exp.cuda.JcudaVectorInscale.*;
import static com.spbsu.exp.cuda.data.DataUtils.contractBottomRow;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:52
 */
public class NeuralNetsLearning {

  private NeuralNets nn;
  private float alpha;
//  private float momentum;
//  private float scalingAlpha;
//  private float weightsPenalty;
//  private float nonSparsityPenalty;
//  private float sparsityTarget;
//  private float dropoutLevel;
  private int epochsNumber;
  private int batchSize;

  public NeuralNetsLearning(
      final @NotNull NeuralNets nn,
      final float alpha,
      final int epochsNumber
  ) {
    this.nn = nn;
    this.alpha = alpha;
//    this.momentum = momentum;
//    this.scalingAlpha = scalingAlpha;
//    this.weightsPenalty = weightsPenalty;
//    this.nonSparsityPenalty = nonSparsityPenalty;
//    this.sparsityTarget = sparsityTarget;
//    this.dropoutLevel = dropoutLevel;
    this.epochsNumber = epochsNumber;
    this.batchSize = nn.batchSize;
  }

  public void batchLearn(final @NotNull FMatrix X, final @NotNull FMatrix Y) {
    final int examplesNumber = X.getColumns();
    final int batchesNumber = examplesNumber / batchSize;
    final int lastLayerIndex = nn.weights.length;

    for (int i = 0; i < epochsNumber; i++) {
      final TIntArrayList examplesIndexes = DataUtils.randomPermutations(examplesNumber);

      for (int j = 0; j < batchesNumber; j++) {
        final TIntList indexes = examplesIndexes.subList(j * batchSize, (j + 1) * batchSize);
        final FMatrix batchX = X.getColumnsRange(indexes);
        final FMatrix batchY = Y.getColumnsRange(indexes);

        final FMatrix output = nn.batchForward(batchX);
        final FMatrix error = fSubtr(batchY, output);

        final FMatrix[] D = backPropagation(error);
        updateWeights(D);
      }
      System.out.println("Epoch " + i);
    }
  }

  private FMatrix[] backPropagation(final FMatrix error) {
    final int lastLayerIndex = nn.weights.length;
    final FMatrix[] D = new FMatrix[lastLayerIndex + 1];
    for (int i = 0; i < lastLayerIndex + 1; i++) {
      final FMatrix activation = nn.activations[i];
      D[i] = new FArrayMatrix(activation.getRows(), activation.getColumns());
    }

    D[lastLayerIndex] = FHadamard(fScale(error, -1.f), (FMatrix)nn.outputRectifier.df(nn.activations[lastLayerIndex]));

    for (int i = lastLayerIndex - 1; i > 0; i--) {
      final FMatrix dA = (FMatrix)nn.rectifier.df(nn.activations[i]);

      //todo(ksen): non sparsity penalty

      if (i + 1 == lastLayerIndex) {
        D[i] = FHadamard(fMult(nn.weights[i], true, D[i + 1], false), dA); //todo(ksen): sparsity error
      }
      else {
        D[i] = FHadamard(fMult(nn.weights[i], true, contractBottomRow(D[i + 1]), false), dA); //todo(ksen): sparsity error
      }

      //todo(ksen): dropout
    }
    for (int i = 0; i < lastLayerIndex; i++) {
      if (i + 1 == lastLayerIndex) {
        D[i] = fScale(fMult(D[i + 1], false, nn.activations[i], true), 1.f / D[i + 1].getColumns());
      }
      else {
        D[i] = fScale(fMult(contractBottomRow(D[i + 1]), false, nn.activations[i], true), 1.f / D[i + 1].getColumns());
      }
    }
    return D;
  }

  private void updateWeights(final FMatrix[] D) {
    final int weightsNumber = nn.weights.length;
    for (int i = 0; i < weightsNumber; i++) {
      //todo(ksen): L2 penalty

      final FMatrix dW = fScale(D[i], alpha);

      //todo(ksen): momentum

      nn.weights[i] = fSubtr(nn.weights[i], dW);
    }
  }

}
