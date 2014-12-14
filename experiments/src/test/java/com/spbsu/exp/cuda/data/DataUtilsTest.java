package com.spbsu.exp.cuda.data;

import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
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

  @Test
  public void testExtendAsBottomRow() throws Exception {
    FMatrix A = new FArrayMatrix(3, new float[]{1, 2, 3, 1, 2, 3, 1, 2, 3});
    final FVector b = new FArrayVector(new float[]{4, 4, 4});

    System.out.println(A);

    A = DataUtils.extendAsBottomRow(A, b);

    System.out.println(A);
  }

  @Test
  public void testContractBottomRow() throws Exception {
    FMatrix A = new FArrayMatrix(3, new float[]{1, 2, 3, 1, 2, 3, 1, 2, 3});

    System.out.println(A);

    A = DataUtils.contractBottomRow(A);

    System.out.println(A);
  }

  @Test
  public void testOnce() throws Exception {
    FVector a = new FArrayVector(new float[]{1, 2, 3});

    final FArrayVector b = (FArrayVector)DataUtils.extendAsBottom(a, 10);

    System.out.println(b.toString(2));

    final FArrayVector c = (FArrayVector)DataUtils.contractBottom(b);

    System.out.println(c.toString(2));
  }
}
