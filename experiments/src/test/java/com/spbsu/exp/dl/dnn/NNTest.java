package com.spbsu.exp.dl.dnn;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.cuda.process.functions.floats.IdenticalFA;
import com.spbsu.exp.cuda.process.functions.floats.SigmoidFA;
import com.spbsu.exp.cuda.process.functions.floats.TanhFA;
import com.spbsu.exp.cuda.process.functions.floats.TanhOptimalFA;
import com.spbsu.exp.dl.Init;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.XChartPanel;

import javax.swing.*;
import java.util.*;
import java.util.Timer;

/**
 * jmll
 * ksen
 * 14.December.2014 at 01:34
 */
public class NNTest {

  public static final String TARGET_SERIES = "Target";
  public static final String RESULT_SERIES = "Result";

  public static void main(String[] args) {
    final Chart chart = new Chart(1000, 700);
    chart.setChartTitle("Sample Real-time Chart");
    chart.setXAxisTitle("X");
    chart.setYAxisTitle("Y");

    final Series targetSeries = chart.addSeries(TARGET_SERIES, new double[]{1}, new double[]{1});
    targetSeries.setMarker(SeriesMarker.NONE);

    final Series resultSeries = chart.addSeries(RESULT_SERIES, new double[]{1}, new double[]{1});
    resultSeries.setMarker(SeriesMarker.NONE);

    final XChartPanel chartPanel = new XChartPanel(chart);

    SwingUtilities.invokeLater(new Runnable() {
      @Override
      public void run() {
        JFrame frame = new JFrame("XChart");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(chartPanel);
        frame.pack();
        frame.setVisible(true);
      }
    });

    final int epochs = 100;
    final int examples = 1000;
    final float learningRate = 0.07f;

    final Random random = new Random(1);
    final float[] data = new float[examples];
    for (int i = 0; i < examples; i++) {
      data[i] = random.nextFloat() * 10;
    }
    Arrays.sort(data);

    final FMatrix X = new FArrayMatrix(1, examples);
    final FMatrix Y = new FArrayMatrix(1, examples);
    for (int i = 0; i < examples; i++) {
      X.set(0, i, data[i]);
      Y.set(0, i, (float)Math.log(data[i]));
    }
    final List<Float> xSeries = new ArrayList<>(examples);
    final List<Float> ySeries = new ArrayList<>(examples);
    for (int i = 0; i < examples; i++) {
      xSeries.add(data[i]);
      ySeries.add((float)Math.log(data[i]));
    }
    chartPanel.updateSeries(TARGET_SERIES, xSeries, ySeries);

    final NeuralNets nn = new NeuralNets(
        new int[]{1, 200, 50, 1},
        100,
        new SigmoidFA(),
        new IdenticalFA(),
        Init.RANDOM_SMALL
    );
    final NeuralNetsLearning nnl = new NeuralNetsLearning(nn, learningRate, 1);

    for (int i = 0; i < epochs; i++) {
      nnl.batchLearn(X, Y);

      final List<Float> result = new ArrayList<>(examples);
      for (int j = 0; j < examples; j++) {
        result.add(nn.forward(X.getColumn(j)).get(0));
      }
      chartPanel.updateSeries(RESULT_SERIES, xSeries, result);
    }
  }

}
