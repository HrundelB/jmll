package com.spbsu.exp.dl.rbm;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
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

  private FMatrix W;
  private FVector b;
  private FVector c;

  public RestrictedBoltzmannMachine(final int vuNumber, final int huNumber) {
    W = new FArrayMatrix(vuNumber, huNumber);
    b = new FArrayVector(vuNumber);
    c = new FArrayVector(huNumber);
  }

  public RestrictedBoltzmannMachine(
      final @NotNull FMatrix W,
      final @NotNull FVector b,
      final @NotNull FVector c
  ) {
    this.W = W;
    this.b = b;
    this.c = c;
  }

  public RestrictedBoltzmannMachine(final @NotNull String path2model) {
    read(path2model);
  }

  public void init(final @NotNull Init init) {
    final int rows = W.getRows();
    final int columns = W.getColumns();

    switch (init) {
      case RANDOM_SMALL: {
        final Random random = new Random();
        final float scale = 0.2f;
        final float shift = scale / 2.f;

        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < columns; j++) {
            W.set(i, j, random.nextFloat() * scale - shift);
          }
        }
        for (int i = 0; i < rows; i++) {
          b.set(i, random.nextFloat() * scale - shift);
        }
        for (int i = 0; i < columns; i++) {
          c.set(i, random.nextFloat() * scale - shift);
        }
        break;
      }
      case IDENTITY: {
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < columns; j++) {
            W.set(i, j, 1.f);
          }
        }
        for (int i = 0; i < rows; i++) {
          b.set(i, 1.f);
        }
        for (int i = 0; i < columns; i++) {
          c.set(i, 1.f);
        }
        break;
      }
    }
  }

  public FMatrix batchPositive(final @NotNull FMatrix X) {
    fSum(fRepeatAsRow(c, X.getRows()), fMult(X, W));
    return null;
  }

  // CD-1
  public void learn(FVector input, float alpha) {
    final FVector h = positive(input);

    final FVector vN = negative(h);
    final FVector hN = positive(vN);

    // W = W + alpha * (v * trans(h) - v' * trans(h'))
    W = fSum(W, fScale(fSubtr(fMult(input, h), fMult(vN, hN)), alpha));
  }

  public FVector represent(final @NotNull FVector input) {
    final FVector h = positive(input);
    return negative(h);
  }

  private FVector positive(FVector v) {
    final FVector z = fMult(v, W);
    FTanh(z);
    return z;
  }

  private FVector negative(FVector vP) {
    final FVector zP = fMult(W, vP);
    FTanh(zP);
    return zP;
  }

  public void read(final @NotNull String path) {
    try(
        final RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, 4 * fileChannel.size());

      final int vuNumber = buffer.getInt();
      final int huNumber = buffer.getInt();

      W = new FArrayMatrix(vuNumber, read(vuNumber * huNumber, buffer));
      b = new FArrayVector(read(vuNumber, buffer));
      c = new FArrayVector(read(huNumber, buffer));
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
    final int vuNumber = W.getRows();
    final int huNumber = W.getColumns();
    final long size = 2 + 4 * (vuNumber * huNumber + vuNumber + huNumber);
    try(
        final RandomAccessFile raf = new RandomAccessFile(new File(path), "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, size);

      buffer.putInt(vuNumber);
      buffer.putInt(huNumber);

      write(W.toArray(), buffer);
      write(b.toArray(), buffer);
      write(c.toArray(), buffer);
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
