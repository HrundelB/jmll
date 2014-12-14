package com.spbsu.exp.dl.dnn;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.process.functions.floats.IdenticalFA;
import com.spbsu.exp.cuda.process.functions.floats.SigmoidFA;
import com.spbsu.exp.cuda.process.functions.ArrayUnaryFunction;
import com.spbsu.exp.cuda.data.DataUtils;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.Init;
import com.spbsu.exp.cuda.data.FMatrix;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

import static com.spbsu.exp.cuda.JcublasHelper.*;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:47
 */
public class NeuralNets {

  public FMatrix[] weights;
  public FMatrix[] activations;
  public FVector[] average;
  public ArrayUnaryFunction<float[]> rectifier;
  public ArrayUnaryFunction<float[]> outputRectifier;
  public int batchSize;

  public NeuralNets(
      final @NotNull int[] layersDimensions,
      final int batchSize,
      final @NotNull ArrayUnaryFunction<float[]> rectifier,
      final @NotNull ArrayUnaryFunction<float[]> outputRectifier,
      final @NotNull Init initMethod
  ) {
    final int length = layersDimensions.length;

    weights = new FMatrix[length - 1];
    activations = new FMatrix[length];

    for (int i = 0; i < length - 1; i++) {
      weights[i] = new FArrayMatrix(layersDimensions[i + 1], layersDimensions[i] + 1);
      activations[i] = new FArrayMatrix(layersDimensions[i], batchSize);
    }
    activations[length - 1] = new FArrayMatrix(layersDimensions[length - 1], batchSize);

    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
    this.batchSize = batchSize;
    init(initMethod);
  }

  public NeuralNets(
      final @NotNull FMatrix[] weights,
      final @NotNull FMatrix[] activations,
      final @NotNull ArrayUnaryFunction<float[]> rectifier,
      final @NotNull ArrayUnaryFunction<float[]> outputRectifier
  ) {
    this.weights = weights;
    this.activations = activations;
    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
  }

  public NeuralNets(final @NotNull String path2model) {
    read(path2model);
  }

  private void init(final @NotNull Init initMethod) {
    switch (initMethod) {
      case RANDOM_SMALL : {
        final Random random = new Random(1);
        final FMatrix[] W = weights;

        for (int i = 0; i < W.length; i++) {
          final FMatrix wights = W[i];

          final int rows = wights.getRows();
          final int columns = wights.getColumns();
          final float scale = 8.f * (float) Math.sqrt(6. / (rows + columns - 1.));
          for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
              wights.set(j, k, (random.nextFloat() - 0.5f) * scale);
            }
          }
        }
        break;
      }
      case DO_NOTHING : {
        break;
      }
    }
  }

  public FVector forward(final @NotNull FVector x) {
    FVector input = DataUtils.extendAsBottom(x, 1.f);
    FVector output;

    final int last = weights.length;
    for (int i = 1; i < last; i++) {
      output = (FVector)rectifier.f(fMult(weights[i - 1], input));
      input = DataUtils.extendAsBottom(output, 1.f);
    }
    return (FVector)outputRectifier.f(fMult(weights[last - 1], input));
  }

  public FMatrix batchForward(final @NotNull FMatrix X) {
    final FVector once = DataUtils.once(X.getColumns());
    activations[0] = DataUtils.extendAsBottomRow(X, once);

    final int last = weights.length;
    for (int i = 1; i < last; i++) {
      activations[i] = (FMatrix)rectifier.f(fMult(weights[i - 1], activations[i - 1]));

      //todo(ksenon): dropout, sparsity

      activations[i] = DataUtils.extendAsBottomRow(activations[i], once);
    }
    activations[last] = (FMatrix)outputRectifier.f(fMult(weights[last - 1], activations[last - 1]));

    return activations[last];
  }

  public void read(final @NotNull String path) {
    try (
        final RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fc = raf.getChannel()
    ) {
      final MappedByteBuffer byteBuffer = fc.map(FileChannel.MapMode.READ_WRITE, 0, fc.size());

      byteBuffer.getInt();
      rectifier = new SigmoidFA();
      byteBuffer.getInt();
      outputRectifier = new IdenticalFA();
      weights = new FMatrix[byteBuffer.getInt()];

      for (int i = 0; i < weights.length; i++) {
        final FMatrix W = new FArrayMatrix(byteBuffer.getInt(), byteBuffer.getInt());

        for (int row = 0; row < W.getRows(); row++) {
          for (int column = 0; column < W.getColumns(); column++) {
            W.set(row, column, byteBuffer.getFloat());
          }
        }
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void write(final @NotNull String path) {
    long size = 3 + 2 * weights.length;
    for (int i = 0; i < weights.length; i++) {
      size += weights[i].getRows() * weights[i].getColumns();
    }
    try (
        final RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fc = raf.getChannel()
    ) {
      final MappedByteBuffer byteBuffer = fc.map(FileChannel.MapMode.READ_WRITE, 0, 4 * size);

      byteBuffer.putInt(0);
      byteBuffer.putInt(1);
      byteBuffer.putInt(weights.length);

      for (int i = 0; i < weights.length; i++) {
        final FMatrix W = weights[i];
        byteBuffer.putInt(W.getRows());
        byteBuffer.putInt(W.getColumns());

        for (int row = 0; row < W.getRows(); row++) {
          for (int column = 0; column < W.getColumns(); column++) {
            byteBuffer.putFloat(W.get(row, column));
          }
        }
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

}
