package com.spbsu.exp.dl.dnn.rectifiers.impl;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.JcublasHelper;
import com.spbsu.exp.cuda.JcudaVectorInscale;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.dl.dnn.rectifiers.Rectifier;

/**
 * jmll
 * ksen
 * 07.December.2014 at 18:29
 */
public class Sigmoid implements Rectifier {

  @Override
  public FMatrix activate(final @NotNull FMatrix Z) {
    return JcudaVectorInscale.fSigmoid(Z);
  }

  @Override
  public FMatrix derivative(@NotNull FMatrix Z) {
    return JcublasHelper.fSubtr(Z, JcudaVectorInscale.FHadamard(Z, Z));
  }

}
