package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import com.spbsu.exp.dl.Init;
import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * jmll
 * ksen
 * 26.October.2014 at 11:29
 */
public class RBMLearningTest extends Assert {

  private String path = "/home/ksen/Documents/data/MNIST/";
  private String lx = "train-images.idx3-ubyte";
  private String ly = "train-labels.idx1-ubyte";
  private String tx = "t10k-images.idx3-ubyte";
  private String ty = "t10k-labels.idx1-ubyte";

  @Test
  public void testLearn() throws Exception {
    final int dDim = 28 * 28;
    final int examples = 1000;

    // LOAD DS
    long begin = System.currentTimeMillis();
    final File learn = new File(path, lx);
    final File learnLabels = new File(path, ly);
    final File test = new File(path, tx);
    final File testLabels = new File(path, ty);
    final FMatrix lX = new FArrayMatrix(784, examples);
    final List<FVector> ly = new ArrayList<>(60000);
    final List<FVector> tx = new ArrayList<>(10000);
    final List<FVector> ty = new ArrayList<>(10000);
    try (
        RandomAccessFile raf = new RandomAccessFile(learn, "rw");
        RandomAccessFile raf1 = new RandomAccessFile(learnLabels, "rw");
        RandomAccessFile raf2 = new RandomAccessFile(test, "rw");
        RandomAccessFile raf3 = new RandomAccessFile(testLabels, "rw");
        final FileChannel fcLearnX = raf.getChannel();
        final FileChannel fcLearnY = raf1.getChannel();
        final FileChannel fcTestX = raf2.getChannel();
        final FileChannel fcTestY = raf3.getChannel()
    ) {
      final MappedByteBuffer bufferLX = fcLearnX.map(FileChannel.MapMode.READ_WRITE, 0, learn.length());
      final MappedByteBuffer bufferLY = fcLearnY.map(FileChannel.MapMode.READ_WRITE, 0, learnLabels.length());
      final MappedByteBuffer bufferTX = fcTestX.map(FileChannel.MapMode.READ_WRITE, 0, test.length());
      final MappedByteBuffer bufferTY = fcTestY.map(FileChannel.MapMode.READ_WRITE, 0, testLabels.length());

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
        lX.setColumn(i, image);
      }

      magic = bufferLY.getInt();
      examplesNumber = bufferLY.getInt();

      assertEquals(2049, magic);
      assertEquals(60000, examplesNumber);

      for (int i = 0; i < examplesNumber; i++) {
        ly.add(toVector(bufferLY.get()));
      }

      magic = bufferTX.getInt();
      examplesNumber = bufferTX.getInt();
      rows = bufferTX.getInt();
      columns = bufferTX.getInt();

      assertEquals(2051, magic);
      assertEquals(10000, examplesNumber);
      assertEquals(28, rows);
      assertEquals(28, columns);

      for (int i = 0; i < examplesNumber; i++) {
        final float[] image = new float[dDim];
        for (int j = 0; j < dDim; j++) {
          image[j] = bufferTX.get() != 0 ? 1.f : 0.f;
        }
        tx.add(new FArrayVector(image));
      }

      magic = bufferTY.getInt();
      examplesNumber = bufferTY.getInt();

      assertEquals(2049, magic);
      assertEquals(10000, examplesNumber);

      for (int i = 0; i < examplesNumber; i++) {
        ty.add(toVector(bufferTY.get()));
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    System.out.println("Data loaded. " + (System.currentTimeMillis() - begin));

    //LEARN
    begin = System.currentTimeMillis();
    final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(784, 1000, 20);
    rbm.init(Init.RANDOM_SMALL);
    final RBMLearning learning = new RBMLearning(rbm, 0.1f, 0.f, 15, 20);

    learning.learn(lX);
    System.out.println("Trained. " + (System.currentTimeMillis() - begin));

    rbm.write("experiments/src/test/data/dl/rbm/rbm.data");
  }

  private void wi(FVector image, String path) throws IOException {
    final float[] img = image.toArray();
    BufferedImage bImageFromConvert = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
    final WritableRaster raster = bImageFromConvert.getRaster();
    for (int i = 0; i < img.length; i++) {
      raster.setSample(i % 28, i / 28, 0, img[i]);
    }

    ImageIO.write(bImageFromConvert, "png", new File(path));
  }

  private FVector toVector(final byte index) {
    final FVector y = new FArrayVector(10);
    y.set(index, 1.f);
    return y;
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
