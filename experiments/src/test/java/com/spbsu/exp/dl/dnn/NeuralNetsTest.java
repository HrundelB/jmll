package com.spbsu.exp.dl.dnn;

import org.junit.Test;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import com.spbsu.exp.dl.dnn.rectifiers.impl.Sigmoid;
import org.junit.Assert;

/**
 * jmll
 * ksen
 * 07.December.2014 at 23:04
 */
public class NeuralNetsTest extends Assert {

  @Test
  public void testStartup() throws Exception {
    final NeuralNets nn = new NeuralNets(
        new int[]{2, 3, 1},
        3,
        new Sigmoid(),
        new Sigmoid(),
        Init.DO_NOTHING
    );

    System.out.println(nn.batchForward(new FArrayMatrix(2, new float[]{1, 2, 1, 2, 1, 2})));
  }
}
