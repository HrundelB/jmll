package com.spbsu.exp.cuda.process.functions;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.ArrayBased;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:14
 */
public interface ArrayUnaryFunction<T> {

  @NotNull ArrayBased<T> f(final @NotNull ArrayBased<T> x);

  @NotNull ArrayBased<T> df(final @NotNull ArrayBased<T> x);

}
