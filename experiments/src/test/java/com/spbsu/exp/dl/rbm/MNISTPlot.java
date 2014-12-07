package com.spbsu.exp.dl.rbm;

import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.dl.Init;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.StringTokenizer;

/**
 * jmll
 * ksen
 * 29.November.2014 at 16:25
 */
public class MNISTPlot {

  public static void main(String[] args) {
    final double[] x = new double[1000];
    final double[][] y = new double[2][1000];
    final String prefix = "experiments/src/test/data/dl/classif-";
    try (
        final LineNumberReader reader1 = new LineNumberReader(new FileReader(prefix + "libfm.txt"));
        final LineNumberReader reader2 = new LineNumberReader(new FileReader(prefix + "rbm+libfm.txt"))
    ) {
      for (int i = 0; i < 1000; i++) {
        final String currentLine1 = reader1.readLine();
        final String currentLine2 = reader2.readLine();
        final StringTokenizer tokenizer1 = new StringTokenizer(currentLine1, "\t");
        final StringTokenizer tokenizer2 = new StringTokenizer(currentLine2, "\t");

        x[i] = i;
        tokenizer1.nextToken();
        tokenizer2.nextToken();

        tokenizer1.nextToken();
        tokenizer1.nextToken();
        tokenizer2.nextToken();
        tokenizer2.nextToken();

        y[0][i] = Double.parseDouble(tokenizer1.nextToken());
        y[1][i] = Double.parseDouble(tokenizer2.nextToken());
//        y[2][i] = Double.parseDouble(tokenizer1.nextToken());
//        y[3][i] = Double.parseDouble(tokenizer2.nextToken());
//        y[4][i] = Double.parseDouble(tokenizer2.nextToken());
//        y[5][i] = Double.parseDouble(tokenizer2.nextToken());
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }

    Chart chart = QuickChart.getChart("Error", "X", "Y",
        new String[]{"Test ll FM", "Test ll RBM+FM"}, x, y);

    new SwingWrapper(chart).displayChart();
  }

}
