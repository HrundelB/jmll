package com.spbsu.ml.io;

import com.spbsu.commons.func.Converter;
import com.spbsu.commons.seq.CharSeqReader;
import com.spbsu.ml.dynamicGridFix.implFix.BFDynamicGrid;
import com.spbsu.ml.dynamicGridFix.implFix.StaticRow;
import com.spbsu.ml.dynamicGridFix.interfacesFix.DynamicGrid;
import com.spbsu.ml.dynamicGridFix.interfacesFix.DynamicRow;
import gnu.trove.list.array.TDoubleArrayList;

import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class BFDynamicGridStringConverter implements Converter<DynamicGrid, CharSequence> {
  @Override
  public DynamicGrid convertFrom(CharSequence source) {
    final List<StaticRow> rows = new ArrayList<StaticRow>(1000);
    final LineNumberReader reader = new LineNumberReader(new CharSeqReader(source));
    String line;
    try {
      final TDoubleArrayList borders = new TDoubleArrayList();
//      int bfIndex = 0;
      int lineIndex = 0;
      while ((line = reader.readLine()) != null) {
        final StringTokenizer tok = new StringTokenizer(line, " \t");
        borders.clear();
        while (tok.hasMoreElements()) {
          borders.add(Double.parseDouble(tok.nextToken()));
        }
        rows.add(new StaticRow(null, lineIndex, borders.toArray()));
//        bfIndex += borders.size();
        lineIndex++;
      }
    } catch (IOException ioe) { //skip
    }
    return new BFDynamicGrid(rows.toArray(new StaticRow[rows.size()]));
  }

  @Override
  public CharSequence convertTo(DynamicGrid grid) {
    StringBuilder builder = new StringBuilder();
    for (int rowIndex = 0; rowIndex < grid.rows(); rowIndex++) {
      final DynamicRow row = grid.row(rowIndex);
      final int rowBfCount = row.size();
      for (int border = 0; border < rowBfCount; border++) {
        builder.append(border > 0 ? "\t" : "");
        builder.append(row.bf(border).condition());
      }
      builder.append("\n");
    }
    return builder;
  }

}
