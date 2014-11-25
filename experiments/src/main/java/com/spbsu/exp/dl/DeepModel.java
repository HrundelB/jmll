package com.spbsu.exp.dl;

import com.spbsu.exp.cuda.data.FVector;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:29
 */
public interface DeepModel {

  FVector predict(final @NotNull FVector x);

  void read(final @NotNull String path2model);

  void write(final @NotNull String path2model);

}
