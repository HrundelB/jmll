package com.spbsu.exp.dl.dnn.rectifiers;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.FMatrix;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:49
 */
public interface Rectifier {

  FMatrix activate(final @NotNull FMatrix Z);

  FMatrix derivative(final @NotNull FMatrix Z);

}
