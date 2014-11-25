package com.spbsu.exp.dl;

import com.spbsu.exp.cuda.data.FVector;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:30
 */
public interface DeepModelSupervisedLearning {

  void learn(final @NotNull FVector x, final @NotNull FVector y);

  void init(final @NotNull Init initMethod);

}
