package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.dl.Init;
import gnu.trove.list.array.TByteArrayList;
import org.junit.Test;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import org.junit.Assert;

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
    final int examples = 60000;

    // LOAD DS
    long begin = System.currentTimeMillis();
    final File learn = new File(path, lx);
    final File learnLabels = new File(path, ly);
    final File test = new File(path, tx);
    final File testLabels = new File(path, ty);

    final FMatrix lX = new FArrayMatrix(784, examples);
    final TByteArrayList ly = new TByteArrayList(60000);
    final FMatrix tX = new FArrayMatrix(784, 10000);
    final TByteArrayList ty = new TByteArrayList(10000);
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

      int magic = bufferLX.getInt();
      int examplesNumber = bufferLX.getInt();
      int rows = bufferLX.getInt();
      int columns = bufferLX.getInt();

      assertEquals(2051, magic);
      assertEquals(60000, examplesNumber);
      assertEquals(28, rows);
      assertEquals(28, columns);

      magic = bufferLY.getInt();
      examplesNumber = bufferLY.getInt();

      assertEquals(2049, magic);
      assertEquals(60000, examplesNumber);

      for (int i = 0; i < examples; i++) {
        final float[] image = new float[dDim];
        for (int j = 0; j < dDim; j++) {
          image[j] = bufferLX.get() != 0 ? 1.f : 0.f;
        }
        lX.setColumn(i, image);
        ly.add(bufferLY.get());
      }

      final MappedByteBuffer bufferTX = fcTestX.map(FileChannel.MapMode.READ_WRITE, 0, test.length());
      final MappedByteBuffer bufferTY = fcTestY.map(FileChannel.MapMode.READ_WRITE, 0, testLabels.length());

      magic = bufferTX.getInt();
      examplesNumber = bufferTX.getInt();
      rows = bufferTX.getInt();
      columns = bufferTX.getInt();

      assertEquals(2051, magic);
      assertEquals(10000, examplesNumber);
      assertEquals(28, rows);
      assertEquals(28, columns);

      magic = bufferTY.getInt();
      examplesNumber = bufferTY.getInt();

      assertEquals(2049, magic);
      assertEquals(10000, examplesNumber);

      for (int i = 0; i < examplesNumber; i++) {
        final float[] image = new float[dDim];
        for (int j = 0; j < dDim; j++) {
          image[j] = bufferTX.get() != 0 ? 1.f : 0.f;
        }
        tX.setColumn(i, image);
        ty.add(bufferTY.get());
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    System.out.println("Data loaded. " + (System.currentTimeMillis() - begin));

    //LEARN
    begin = System.currentTimeMillis();
    final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(784, 500, 1000, Init.DO_NOTHING);
    final RBMLearning learning = new RBMLearning(rbm, 0.1f, 0.f, 15, 1000);

    learning.learn(lX);
    System.out.println("Trained. " + (System.currentTimeMillis() - begin));

    //REPRESENT
    try (
        final FileWriter libfmTrain = new FileWriter("experiments/src/test/data/dl/mnist-train-rbm.libfm");
        final FileWriter libfmTest = new FileWriter("experiments/src/test/data/dl/mnist-test-rbm.libfm")
    ) {
      final FMatrix libfmInputTrain = rbm.batchPositive(lX);
      for (int i = 0; i < libfmInputTrain.getColumns(); i++) {
        libfmTrain.write(ly.get(i) + " ");

        final FVector input = libfmInputTrain.getColumn(i);
        for (int j = 0; j < input.getDimension(); j++) {
          if (input.get(j) != 0) {
            libfmTrain.write(j + ":" + 1 + " ");

          }
        }
        libfmTrain.write('\n');
      }

      final FMatrix libfmInputTest = rbm.batchPositive(tX);
      for (int i = 0; i < libfmInputTest.getColumns(); i++) {
        libfmTest.write(ty.get(i) + " ");

        final FVector input = libfmInputTest.getColumn(i);
        for (int j = 0; j < input.getDimension(); j++) {
          if (input.get(j) != 0) {
            libfmTest.write(j + ":" + 1 + " ");

          }
        }
        libfmTest.write('\n');
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    rbm.write("experiments/src/test/data/dl/rbm/rbm.data");
  }

  @Test
  public void testName() throws Exception {
    final int dDim = 28 * 28;

    final File learn = new File(path, lx);
    final File learnLabels = new File(path, ly);
    final File test = new File(path, tx);
    final File testLabels = new File(path, ty);
    try (
        final RandomAccessFile raf = new RandomAccessFile(learn, "rw");
        final RandomAccessFile raf1 = new RandomAccessFile(learnLabels, "rw");
        final RandomAccessFile raf2 = new RandomAccessFile(test, "rw");
        final RandomAccessFile raf3 = new RandomAccessFile(testLabels, "rw");
        final FileChannel fcLearnX = raf.getChannel();
        final FileChannel fcLearnY = raf1.getChannel();
        final FileChannel fcTestX = raf2.getChannel();
        final FileChannel fcTestY = raf3.getChannel();
        final FileWriter libfmTrain = new FileWriter("experiments/src/test/data/dl/mnist-train.libfm");
        final FileWriter libfmTest = new FileWriter("experiments/src/test/data/dl/mnist-test.libfm")
    ) {
      final MappedByteBuffer bufferLX = fcLearnX.map(FileChannel.MapMode.READ_WRITE, 0, learn.length());
      final MappedByteBuffer bufferLY = fcLearnY.map(FileChannel.MapMode.READ_WRITE, 0, learnLabels.length());

      int magic = bufferLX.getInt();
      int examplesNumber = bufferLX.getInt();
      int rows = bufferLX.getInt();
      int columns = bufferLX.getInt();

      assertEquals(2051, magic);
      assertEquals(60000, examplesNumber);
      assertEquals(28, rows);
      assertEquals(28, columns);

      magic = bufferLY.getInt();
      examplesNumber = bufferLY.getInt();

      assertEquals(2049, magic);
      assertEquals(60000, examplesNumber);

      for (int i = 0; i < examplesNumber; i++) {
        libfmTrain.write(bufferLY.get() + " ");

        for (int j = 0; j < dDim; j++) {
          int value;
          if ((value = bufferLX.get()) != 0) {
            libfmTrain.write(j + ":" + 1 + " ");
          }
        }
        libfmTrain.write('\n');
      }

      final MappedByteBuffer bufferTX = fcTestX.map(FileChannel.MapMode.READ_WRITE, 0, test.length());
      final MappedByteBuffer bufferTY = fcTestY.map(FileChannel.MapMode.READ_WRITE, 0, testLabels.length());

      magic = bufferTX.getInt();
      examplesNumber = bufferTX.getInt();
      rows = bufferTX.getInt();
      columns = bufferTX.getInt();

      assertEquals(2051, magic);
      assertEquals(10000, examplesNumber);
      assertEquals(28, rows);
      assertEquals(28, columns);

      magic = bufferTY.getInt();
      examplesNumber = bufferTY.getInt();

      assertEquals(2049, magic);
      assertEquals(10000, examplesNumber);

      for (int i = 0; i < examplesNumber; i++) {
        libfmTest.write(bufferTY.get() + " ");

        for (int j = 0; j < dDim; j++) {
          int value;
          if ((value = bufferTX.get()) != 0) {
            libfmTest.write(j + ":" + 1 + " ");
          }
        }
        libfmTest.write('\n');
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private void wi(FVector image, String path) {
    final float[] img = image.toArray();
    BufferedImage bImageFromConvert = new BufferedImage(28, 28, BufferedImage.TYPE_USHORT_GRAY);
    final WritableRaster raster = bImageFromConvert.getRaster();
    for (int i = 0; i < img.length; i++) {
      raster.setSample(i % 28, i / 28, 0, img[i]);
    }

    try {
      ImageIO.write(bImageFromConvert, "png", new File(path));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
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
