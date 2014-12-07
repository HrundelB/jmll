package com.spbsu.exp.dl.dnn.rectifiers.impl;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.DataUtils;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.dl.dnn.rectifiers.Rectifier;

/**
 * jmll
 * ksen
 * 07.December.2014 at 23:15
 */
public class Linear implements Rectifier {

  @Override
  public FMatrix activate(final @NotNull FMatrix Z) {
    return Z;
  }

  @Override
  public FMatrix derivative(final @NotNull FMatrix Z) {
    return DataUtils.once(Z.getRows(), Z.getColumns());
  }

}
