package com.spbsu.exp.dl.dnn.examples;

import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import com.spbsu.exp.dl.dnn.NeuralNets;
import gnu.trove.list.array.TIntArrayList;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;

public class NNUI extends JDialog implements MouseMotionListener {

  private JPanel contentPane;
  private JButton buttonClassify;
  private JButton buttonClear;
  private JList list;
  private JLabel statusLabel;
  private JPanel drawingPanel;

  private NeuralNets neuralNets;
  private DefaultListModel<CharSequence> listModel;

  private boolean pressed = false;
  private TIntArrayList dots = new TIntArrayList(1000);

  public NNUI() {
    setContentPane(contentPane);
    setModal(true);
    getRootPane().setDefaultButton(buttonClassify);
    listModel = new DefaultListModel<>();
    list.setModel(listModel);

    buttonClassify.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        onClassify();
      }
    });

    buttonClear.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        onClear();
      }
    });


    addMouseListener(new MouseAdapter() {
      public void mousePressed(MouseEvent evt) {
        pressed = true;
      }

      public void mouseClicked(MouseEvent evt) {
        pressed = false;
      }
    });
    addMouseMotionListener(this);

    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
  }

  private void onClassify() {
    try {
      final BufferedImage formImage = new BufferedImage(
          drawingPanel.getWidth(),
          drawingPanel.getHeight(),
          BufferedImage.TYPE_BYTE_GRAY
      );
      final Graphics2D formInput = formImage.createGraphics();
      paintAll(formInput);
      formInput.dispose();

      final BufferedImage scaledImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
      final Graphics2D scaledInput = scaledImage.createGraphics();
      scaledInput.drawImage(formImage, 0, 0, 28, 28, null);
      scaledInput.dispose();

      final Raster data = scaledImage.getData();
      final float[] nnInput = new float[28 * 28];
      for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
          nnInput[i + j * 28] = data.getSample(i, j, 0);
        }
      }

      final FVector y = neuralNets.forward(new FArrayVector(nnInput));

      for (int i = 0; i < y.getDimension(); i++) {
        listModel.addElement("p(" + i + ") = " + y.get(i));
      }
    }
    catch(Exception exception) {
      statusLabel.setText(exception.getMessage());
    }
  }

  private void onClear() {
    final Graphics g = getGraphics();
    g.setColor(Color.black);
    for (int i = 0; i < dots.size(); i += 2) {
      g.fillOval(dots.get(i), dots.get(i + 1), 30, 30);
    }
    dots.clear();
    listModel.clear();
  }

  public void paint(Graphics g) {
    if (pressed) {
      g.setColor(Color.white);
      for (int i = 0; i < dots.size(); i += 2) {
        g.fillOval(dots.get(i), dots.get(i + 1), 30, 30);
      }
    }
  }

  public void mouseMoved(MouseEvent evt) {
  }

  public void mouseDragged(MouseEvent evt) {
    dots.add((int) evt.getPoint().getX());
    dots.add((int) evt.getPoint().getY());
    repaint();
  }

  private void loadModel() {
    new SwingWorker<Double, Double>() {
      @Override
      protected Double doInBackground() throws Exception {
        neuralNets = new NeuralNets("experiments/src/test/data/dl/dnn.data");
        statusLabel.setText("Ready.");
        return 0.;
      }
    }.execute();
  }

  public static void main(String[] args) {
    NNUI dialog = new NNUI();
    dialog.pack();
    dialog.loadModel();
    dialog.setVisible(true);
    System.exit(0);
  }

}
