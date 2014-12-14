package com.spbsu.exp.cuda.process.functions;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.ArrayBased;

/**
 * jmll
 * ksen
 * 10.December.2014 at 01:08
 */
public interface ArrayBinaryFunction<T> {

  @NotNull ArrayBased<T> f(final @NotNull ArrayBased<T> x, final @NotNull ArrayBased<T> z);

}
