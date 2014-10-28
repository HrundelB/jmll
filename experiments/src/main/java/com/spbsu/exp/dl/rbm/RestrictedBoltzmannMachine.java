package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.dl.cuda.data.FMatrix;
import com.spbsu.exp.dl.cuda.data.FVector;
import com.spbsu.exp.dl.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.cuda.data.impl.FArrayVector;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

import static com.spbsu.exp.dl.cuda.JcublasHelper.*;
import static com.spbsu.exp.dl.cuda.JcudaVectorInscale.*;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:43
 */
public class RestrictedBoltzmannMachine {

  private FMatrix W;
  private FVector h;
  private FVector a;
  private FVector b;

  public RestrictedBoltzmannMachine(final int vuNumber, final int huNumber) {
    W = new FArrayMatrix(vuNumber, huNumber);
    h = new FArrayVector(huNumber);
    a = new FArrayVector(vuNumber);
    b = new FArrayVector(huNumber);
  }

  public void init(final @NotNull Init init) {
    switch (init) {
      case RANDOM_SMALL: {
        final Random random = new Random();
        final float scale = 0.2f;
        final float shift = scale / 2.f;
        for (int i = 0; i < W.getRows(); i++) {
          for (int j = 0; j < W.getColumns(); j++) {
            W.set(i, j, random.nextFloat() * scale - shift);
          }
        }
        break;
      }
      case IDENTITY: {
        for (int i = 0; i < W.getRows(); i++) {
          for (int j = 0; j < W.getColumns(); j++) {
            W.set(i, j, 1.f);
          }
        }
        break;
      }
    }
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

  public void read(final @NotNull String path) {
    try(
        RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, 4 * (W.getRows() * W.getColumns() + 2));

      W = new FArrayMatrix(buffer.getInt(), buffer.getInt());

      for (int i = 0; i < W.getColumns(); i++) {
        for (int j = 0; j < W.getRows(); j++) {
          W.set(j, i, buffer.getFloat());
        }
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void write(final @NotNull String path) {
    try(
        RandomAccessFile raf = new RandomAccessFile(new File(path), "rw");
        final FileChannel fileChannel = raf.getChannel()
    ) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, 4 * (W.getRows() * W.getColumns() + 2));

      buffer.putInt(W.getRows());
      buffer.putInt(W.getColumns());

      final float[] floats = W.toArray();
      for (int i = 0; i < floats.length; i++) {
        buffer.putFloat(floats[i]);
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
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

  private float energy(final FVector v, final FVector h) {   // +regularization sum(log z)
    return -fDot(a, v) - fDot(b, h) - fDot(fMult(v, W), h);
  }

  public enum Init {
    RANDOM_SMALL,
    IDENTITY
  }

}
