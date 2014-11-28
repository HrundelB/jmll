package com.spbsu.exp.cuda.data;

import org.junit.Test;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import org.junit.Assert;

/**
 * jmll
 * ksen
 * 28.November.2014 at 14:07
 */
public class DataUtilsTest extends Assert {

  @Test
  public void testRepeatAsColumn() throws Exception {
    final FVector b = new FArrayVector(new float[]{0, 1, 2, 3});

    final FMatrix A = DataUtils.repeatAsColumns(b, 3);

    for (int i = 0; i < A.getRows(); i++) {
      for (int j = 0; j < A.getColumns(); j++) {
        assertTrue(i == A.get(i, j));
      }
    }
  }
}
