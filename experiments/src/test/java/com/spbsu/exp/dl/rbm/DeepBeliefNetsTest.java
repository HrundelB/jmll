package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.dl.Init;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.dbn.DeepBeliefNets;
import com.spbsu.exp.dl.dbn.DeepBeliefNetsLearning;
import gnu.trove.list.array.TIntArrayList;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * jmll
 * ksen
 * 21.November.2014 at 01:14
 */
public class DeepBeliefNetsTest extends Assert {

  private String path = "/home/ksen/Documents/data/MNIST/";
  private String lx = "train-images.idx3-ubyte";
  private String ly = "train-labels.idx1-ubyte";
  private String tx = "t10k-images.idx3-ubyte";
  private String ty = "t10k-labels.idx1-ubyte";

  @Test
  public void testMNIST() throws Exception {
    long begin = System.currentTimeMillis();

    // LOAD DS
    final int dDim = 28 * 28;
    final int epochs = 15;
    final int examples = 1000;
    final File learn = new File(path, lx);
    final FMatrix X = new FArrayMatrix(dDim, examples);
    try (
        final RandomAccessFile raf = new RandomAccessFile(learn, "rw");
        final FileChannel fcLearnX = raf.getChannel()
    ) {
      final MappedByteBuffer bufferLX = fcLearnX.map(FileChannel.MapMode.READ_WRITE, 0, learn.length());

      int magic = bufferLX.getInt();
      int examplesNumber = bufferLX.getInt();
      int rows = bufferLX.getInt();
      int columns = bufferLX.getInt();

      assertEquals(2051, magic);
      assertEquals(60000, examplesNumber);
      assertEquals(28, rows);
      assertEquals(28, columns);

      for (int i = 0; i < examples; i++) {
        final float[] image = new float[dDim];
        for (int j = 0; j < dDim; j++) {
          image[j] = bufferLX.get() != 0 ? 1.f : 0.f;
        }
        X.setColumn(i, image);
      }
      System.out.println("Data loaded. " + (System.currentTimeMillis() - begin));
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }

    // LEARN
    begin = System.currentTimeMillis();
    final DeepBeliefNets dbn = new DeepBeliefNets(
        new FMatrix[]{
            new FArrayMatrix(dDim, 500),
            new FArrayMatrix(500, 500),
            new FArrayMatrix(500, 2000),
            new FArrayMatrix(2000, 10)
        }
    );
    final DeepBeliefNetsLearning dbnl = new DeepBeliefNetsLearning(dbn, 0.1f, epochs);
    dbnl.init(Init.RANDOM_SMALL);

    dbnl.learn(X);

    System.out.println("Trained. " + (System.currentTimeMillis() - begin));

    // TEST
    // DS
    begin = System.currentTimeMillis();
    final File test = new File(path, ly);
    final TIntArrayList y = new TIntArrayList(examples);
    try (
        final RandomAccessFile raf = new RandomAccessFile(test, "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final MappedByteBuffer bufferLy = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, test.length());

      int magic = bufferLy.getInt();
      int examplesNumber = bufferLy.getInt();

      assertEquals(2049, magic);
      assertEquals(60000, examplesNumber);

      for (int i = 0; i < examples; i++) {
        y.add(bufferLy.get());
      }
      System.out.println("Test Data loaded. " + (System.currentTimeMillis() - begin));
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }

    // CHECK
    begin = System.currentTimeMillis();
    int counter = 0;
    for (int i = 0; i < examples; i++) {
      final int answer = max(dbn.forward(X.getColumn(i)));
      if(answer != y.get(i)) {
        counter++;
      }
    }
    System.out.println("Result: " + (counter / (examples + 0.f)));
  }

  private int max(FVector fVector) {
    float max = Float.MIN_VALUE;
    int index = 0;
    for (int i = 0; i < fVector.getDimension(); i++) {
      if (fVector.get(i) > max) {
        max = fVector.get(i);
        index = i;
      }
    }
    return index;
  }

}
