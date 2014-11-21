package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.CherryPick;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.CherryRegion;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDCherryRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  private final FastRandom rand = new FastRandom();
  private final double alpha;
  private final double beta;
  private final int maxFailed;

  public GreedyTDCherryRegion(BFGrid grid) {
    this(grid, 0.02, 0.5, 1);
  }

  public GreedyTDCherryRegion(BFGrid grid, double alpha, double beta, int maxFailed) {
    this.grid = grid;
    this.alpha = alpha;
    this.beta = beta;
    this.maxFailed = maxFailed;
  }

  public GreedyTDCherryRegion(BFGrid grid, double alpha, double beta) {
    this(grid, alpha, beta, 1);
  }


  @Override
  public CherryRegion fit(final VecDataSet learn, final Loss loss) {
    final List<BitSet> conditions = new ArrayList<>(100);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    CherryPick pick = new CherryPick(bds, loss.statsFactory());
    int[] points = ArrayTools.sequence(0, learn.length());

    double currentScore = Double.POSITIVE_INFINITY;
    while (true) {
      Pair<BitSet, int[]> result = pick.build(new Evaluator<AdditiveStatistics>() {
        @Override
        public double value(AdditiveStatistics stat) {
          return loss.score(stat);
        }
      }, points, 2);

      if (currentScore <= pick.currentScore + 1e-9) {
        break;
      }

      points = result.getSecond();
      conditions.add(result.getFirst());
      currentScore = pick.currentScore;
    }
    return new CherryRegion(conditions.toArray(new BitSet[conditions.size()]),loss.bestIncrement(pick.inside),grid);
  }


}
