package com.spbsu.exp.dl.datasets;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * jmll
 * ksen
 * 07.December.2014 at 23:31
 */
public class MNIST {

  private static final String PATH = "/home/ksen/Documents/data/MNIST/";
  private static final String LX = "train-images.idx3-ubyte";
  private static final String LY = "train-labels.idx1-ubyte";
  private static final String TX = "t10k-images.idx3-ubyte";
  private static final String TY = "t10k-labels.idx1-ubyte";

  public static FMatrix getTrainDigits(final int examples) {
    final File learn = new File(PATH, LX);

    final int dDim = 784;
    final FMatrix lX = new FArrayMatrix(dDim, examples);
    try (
        final RandomAccessFile raf = new RandomAccessFile(learn, "rw");
        final FileChannel fc = raf.getChannel();
    ) {
      final MappedByteBuffer bufferLX = fc.map(FileChannel.MapMode.READ_WRITE, 0, learn.length());

      int magic = bufferLX.getInt();
      int examplesNumber = bufferLX.getInt();
      int rows = bufferLX.getInt();
      int columns = bufferLX.getInt();

      if (magic != 2051 || examplesNumber != 60000 || rows != 28 || columns != 28) {
        throw new RuntimeException("Something wrong");
      }

      for (int i = 0; i < examples; i++) {
        final float[] image = new float[dDim];
        for (int j = 0; j < dDim; j++) {
          image[j] = bufferLX.get() & 0xFF;
        }
        lX.setColumn(i, image);
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    return lX;
  }

  public static FMatrix getTrainLabels(final int examples) {
    final File learn = new File(PATH, LY);

    final FMatrix lY = new FArrayMatrix(10, examples);
    try (
        final RandomAccessFile raf = new RandomAccessFile(learn, "rw");
        final FileChannel fcLearnX = raf.getChannel();
    ) {
      final MappedByteBuffer bufferLY = fcLearnX.map(FileChannel.MapMode.READ_WRITE, 0, learn.length());

      final int magic = bufferLY.getInt();
      final int examplesNumber = bufferLY.getInt();

      if (magic != 2049 || examplesNumber != 60000) {
        throw new RuntimeException("Something wrong");
      }

      for (int i = 0; i < examples; i++) {
        final byte label = bufferLY.get();
        lY.set(label, i, 1.f);
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    return lY;
  }

  public static FMatrix getTestDigits(final int examples) {
    final File test = new File(PATH, TX);

    final int dDim = 784;
    final FMatrix tX = new FArrayMatrix(dDim, examples);
    try (
        final RandomAccessFile raf = new RandomAccessFile(test, "rw");
        final FileChannel fc = raf.getChannel();
    ) {
      final MappedByteBuffer bufferTX = fc.map(FileChannel.MapMode.READ_WRITE, 0, test.length());

      int magic = bufferTX.getInt();
      int examplesNumber = bufferTX.getInt();
      int rows = bufferTX.getInt();
      int columns = bufferTX.getInt();

      if (magic != 2051 || examplesNumber != 10000 || rows != 28 || columns != 28) {
        throw new RuntimeException("Something wrong");
      }

      for (int i = 0; i < examples; i++) {
        final float[] image = new float[dDim];
        for (int j = 0; j < dDim; j++) {
          image[j] = bufferTX.get() & 0xFF;
        }
        tX.setColumn(i, image);
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    return tX;
  }


  public static FMatrix getTestLabels(final int examples) {
    final File test = new File(PATH, TY);

    final FMatrix tY = new FArrayMatrix(10, examples);
    try (
        final RandomAccessFile raf = new RandomAccessFile(test, "rw");
        final FileChannel fcLearnX = raf.getChannel();
    ) {
      final MappedByteBuffer bufferTY = fcLearnX.map(FileChannel.MapMode.READ_WRITE, 0, test.length());

      final int magic = bufferTY.getInt();
      final int examplesNumber = bufferTY.getInt();

      if (magic != 2049 || examplesNumber != 10000) {
        throw new RuntimeException("Something wrong");
      }

      for (int i = 0; i < examples; i++) {
        final byte label = bufferTY.get();
        tY.set(label, i, 1.f);
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
    return tY;
  }

}
