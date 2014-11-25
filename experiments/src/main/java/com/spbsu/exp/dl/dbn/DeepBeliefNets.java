package com.spbsu.exp.dl.dbn;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import org.jetbrains.annotations.NotNull;

import static com.spbsu.exp.cuda.JcublasHelper.fMult;
import static com.spbsu.exp.cuda.JcudaVectorInscale.FHeaviside;

/**
 * jmll
 * ksen
 * 20.November.2014 at 23:42
 */
public class DeepBeliefNets {

  public FMatrix[] weights;

  public DeepBeliefNets(final FMatrix[] weights) {
    this.weights = weights;
  }

  public FVector forward(final @NotNull FVector x) {
    FVector h = FHeaviside(fMult(x, weights[0]));

    for (int i = 1; i < weights.length; i++) {
      h = FHeaviside(fMult(h, weights[i]));
    }

    return h;
  }


}
