package com.spbsu.exp.dl.dnn.rectifiers;

import com.spbsu.exp.cuda.data.FMatrix;
import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.FVector;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:49
 */
public interface Rectifier {

  FVector activate(final @NotNull FVector z);

  FMatrix activate(final @NotNull FMatrix Z);

}
