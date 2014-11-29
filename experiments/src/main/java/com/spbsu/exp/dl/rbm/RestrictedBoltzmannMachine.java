package com.spbsu.exp.dl.rbm;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.DataUtils;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

import static com.spbsu.exp.cuda.JcublasHelper.*;
import static com.spbsu.exp.cuda.JcudaVectorInscale.*;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:43
 */
public class RestrictedBoltzmannMachine {

  public FMatrix W;
  public FMatrix B;
  public FMatrix C;

  public RestrictedBoltzmannMachine(
      final int visualDim,
      final int hiddenDim,
      final int batchesNumber,
      final @NotNull Init initMethod
  ) {
    this(
        new FArrayMatrix(visualDim, hiddenDim),
        new FArrayMatrix(visualDim, batchesNumber),
        new FArrayMatrix(hiddenDim, batchesNumber),
        initMethod
    );
  }

  public RestrictedBoltzmannMachine(
      final @NotNull FMatrix W,
      final @NotNull FMatrix B,
      final @NotNull FMatrix C,
      final @NotNull Init initMethod
  ) {
    this.W = W;
    this.B = B;
    this.C = C;
    init(initMethod);
  }

  public RestrictedBoltzmannMachine(
      final @NotNull String path2model,
      final int batchSize
  ) {
    read(path2model, batchSize);
  }

  private void init(final Init init) {
    final int n = W.getRows();
    final int k = W.getColumns();
    final int m = C.getRows();

    switch (init) {
      case RANDOM_SMALL: {
        final Random random = new Random();
        final float scale = 0.2f;
        final float shift = scale / 2.f;

        for (int i = 0; i < n; i++) {
          for (int j = 0; j < k; j++) {
            W.set(i, j, random.nextFloat() * scale - shift);
          }
        }

        float[] vector = new float[n];
        for (int i = 0; i < n; i++) {
          vector[i] = random.nextFloat() * scale - shift;
        }
        B = DataUtils.repeatAsColumns(vector, m);

        vector = new float[k];
        for (int i = 0; i < k; i++) {
          vector[i] = random.nextFloat() * scale - shift;
        }
        C = DataUtils.repeatAsColumns(vector, m);
        break;
      }
      case DO_NOTHING : {
        break;
      }
    }
  }

  /**
   *  I{sigmoid(trans(W)[k x n] * X[n x m] + C[k x m]) < U(0, 1)[k x m]}[k x m]
   */
  public FMatrix batchPositive(final @NotNull FMatrix X) {
    return fRndSigmoid(fSum(fMult(W, true, X, false), C));
  }

  /**
   *  I{sigmoid(W[n x k] * H[k x m] + B[n x m]) < U(0, 1)[n x m]}[n x m]
   */
  public FMatrix batchNegative(final @NotNull FMatrix H) {
    return fRndSigmoid(fSum(fMult(W, H), B));
  }

  /**
   *  sigmoid(trans(W)[k x n] * X[n x m] + C[k x m])[k x m]}
   * */
  public FMatrix batchForward(final @NotNull FMatrix X) {
    return fSigmoid(fSum(fMult(W, true, X, false), C));
  }

  public void read(final @NotNull String path, final int batchSize) {
    try (
        final RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, 4 * fileChannel.size());

      final int n = buffer.getInt();
      final int k = buffer.getInt();

      W = new FArrayMatrix(n, read(n * k, buffer));
      B = DataUtils.repeatAsColumns(read(n, buffer), batchSize);
      C = DataUtils.repeatAsColumns(read(k, buffer), batchSize);
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private float[] read(final int size, final ByteBuffer buffer) {
    final float[] array = new float[size];
    for (int i = 0; i < size; i++) {
      array[i] = buffer.getFloat();
    }
    return array;
  }

  public void write(final @NotNull String path) {
    final int n = W.getRows();
    final int k = W.getColumns();
    final long size = 4 * (2 + n * k + n + k);
    try (
        final RandomAccessFile raf = new RandomAccessFile(new File(path), "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, size);

      buffer.putInt(n);
      buffer.putInt(k);

      write(W.toArray(), buffer);
      write(B.getColumn(0).toArray(), buffer);
      write(C.getColumn(0).toArray(), buffer);
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private void write(final float[] array, final ByteBuffer buffer) {
    for (int i = 0; i < array.length; i++) {
      buffer.putFloat(array[i]);
    }
  }

}
