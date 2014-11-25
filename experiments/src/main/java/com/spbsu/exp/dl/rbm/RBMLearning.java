package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.cuda.data.FMatrix;
import gnu.trove.list.array.TIntArrayList;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.Random;

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
      final int epochsNumber,
      final int batchSize
  ) {
    this.rbm = rbm;
    this.alpha = alpha;
    this.momentum = momentum;
    this.epochsNumber = epochsNumber;
    this.batchSize = batchSize;
  }

  public void learn(final @NotNull FMatrix X) {
    final int examplesNumber = X.getColumns();
    final int batchesNumber = examplesNumber / batchSize;

    for (int i = 0; i < epochsNumber; i++) {
      final float error;
      final TIntArrayList examplesIndexes = randomPermutations(examplesNumber);

      for (int j = 0; j < batchesNumber; j++) {
        final FMatrix V0 = X.getColumnsRange(j * batchSize, (j + 1) * batchSize);


      }
    }
  }

  private TIntArrayList randomPermutations(final int size) {
    final Random random = new Random();
    final TIntArrayList list = new TIntArrayList(size);

    for (int i = 0; i < size; i++) {
      list.add(i);
    }
    list.shuffle(random);

    return list;
  }

}
