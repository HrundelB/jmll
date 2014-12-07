package com.spbsu.exp.dl.rbm;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.DataUtils;
import com.spbsu.exp.cuda.data.FMatrix;
import gnu.trove.list.array.TIntArrayList;

import java.util.Random;

import static com.spbsu.exp.cuda.JcublasHelper.*;

/**
 * jmll
 * ksen
 * 25.November.2014 at 14:53
 */
public class RBMLearning {

  private RestrictedBoltzmannMachine rbm;
  private float alpha;
  private float momentum;
  private int epochsNumber;
  private int batchSize;

  public RBMLearning(
      final @NotNull RestrictedBoltzmannMachine rbm,
      final float alpha,
      final float momentum,
      final int epochsNumber
  ) {
    this.rbm = rbm;
    this.alpha = alpha;
    this.momentum = momentum;
    this.epochsNumber = epochsNumber;
    this.batchSize = rbm.B.getColumns();
  }

  public void learn(final @NotNull FMatrix X) {
    final int examplesNumber = X.getColumns();
    final int batchesNumber = examplesNumber / batchSize;

    final float learningRate = alpha / batchSize;
    final float learningMoment = 1.f + momentum;
    for (int i = 0; i < epochsNumber; i++) {
      float error = 0;
      final TIntArrayList examplesIndexes = DataUtils.randomPermutations(examplesNumber);

      for (int j = 0; j < batchesNumber; j++) {
        final FMatrix V0 = X.getColumnsRange(examplesIndexes.subList(j * batchSize, (j + 1) * batchSize));
        final FMatrix H0 = rbm.batchPositive(V0);
        final FMatrix V1 = rbm.batchNegative(H0);
        final FMatrix H1 = rbm.batchPositive(V1);//todo(ksenon): sigm instead of rndsigm

        rbm.W = fSum(
            fScale(rbm.W, learningMoment),
            fScale(fSubtr(fMult(V0, false, H0, true), fMult(V1, false, H1, true)), learningRate)
        );
        rbm.B = fSum(
            fScale(rbm.B, learningMoment),
            fScale(fSubtr(V0, V1), learningRate)
        );
        rbm.C = fSum(
            fScale(rbm.C, learningMoment),
            fScale(fSubtr(H0, H1), learningRate)
        );
        error += DataUtils.rmse(V0, V1) / batchSize;
      }
      System.out.println("Epoch " + i + ", error " + error);
    }
  }

}
