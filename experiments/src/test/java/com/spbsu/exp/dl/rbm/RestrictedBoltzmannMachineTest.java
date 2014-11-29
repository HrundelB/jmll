package com.spbsu.exp.dl.rbm;

import org.junit.Test;
import com.spbsu.commons.system.RuntimeUtils;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import org.junit.Assert;

import java.io.File;
import java.util.Arrays;

/**
 * jmll
 * ksen
 * 25.November.2014 at 15:15
 */
public class RestrictedBoltzmannMachineTest extends Assert {

  @Test
  public void testReadWrite() throws Exception {
    final FArrayMatrix W = new FArrayMatrix(2, new float[]{0, 1, 2, 3, 4, 5, 6, 7});
    final FArrayMatrix B = new FArrayMatrix(3, new float[]{10, 11, 12, 10, 11, 12});
    final FArrayMatrix C = new FArrayMatrix(2, new float[]{20, 21, 20, 21});
    final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(
        W,
        B,
        C,
        Init.DO_NOTHING
    );

    final String tmpDir = RuntimeUtils.getSysTmpDir();
    final String modelPath = new File(tmpDir, "rbm_model_" + System.currentTimeMillis()).getAbsolutePath();

    rbm.write(modelPath);

    final RestrictedBoltzmannMachine rbm2 = new RestrictedBoltzmannMachine(modelPath, 2);

    assertTrue(Arrays.equals(rbm.W.toArray(), rbm2.W.toArray()));
    assertTrue(Arrays.equals(rbm.B.toArray(), rbm2.B.toArray()));
    assertTrue(Arrays.equals(rbm.C.toArray(), rbm2.C.toArray()));
  }

}
