package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import org.junit.Assert;
import org.junit.Test;

/**
 * jmll
 * ksen
 * 25.November.2014 at 15:15
 */
public class RestrictedBoltzmannMachineTest extends Assert {

  @Test
  public void testReadWrite() throws Exception {
    final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(1000, 1001, 1002);

    rbm.init(Init.RANDOM_SMALL);

    rbm.batchNegative(rbm.batchPositive(new FArrayMatrix(1000, 1002)));
  }

}
