package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import com.spbsu.exp.dl.Init;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * jmll
 * ksen
 * 28.October.2014 at 22:30
 */
public class RBMTest {

  public static void main(String[] args) {
    final int dim = 20;
    final int epochs = 15;
    final int examples = 100;
    final int hiddenUnits = 1000;
    final float learningRate = 0.1f;

    final Random random = new Random();
    final List<FVector> ds = new ArrayList<>(examples);
    for (int i = 0; i < examples; i++) {
      float[] data = new float[dim];
      for (int j = 0; j < dim; j++) {
        data[j] = (float) Math.sin(random.nextFloat() + j);
      }
      ds.add(new FArrayVector(data));
    }

    final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(dim, hiddenUnits);
    rbm.init(Init.IDENTITY);

    for (int i = 0; i < epochs; i++) {
      for (int j = 0; j < ds.size(); j++) {
        rbm.learn(ds.get(j), learningRate);
      }
      System.out.println(i);
    }

    final double[] xData = new double[dim];
    final double[][] y = new double[2][dim];
    final FVector represent = rbm.represent(ds.get(0));
    for (int i = 0; i < represent.getDimension(); i++) {
      xData[i] = i;
      y[0][i] = Math.sin(i);
      y[1][i] = represent.get(i);
    }

    Chart chart = QuickChart.getChart("Sin(x)", "X", "Y", new String[]{"Exp", "Act"}, xData, y);

    new SwingWrapper(chart).displayChart();
  }

}
