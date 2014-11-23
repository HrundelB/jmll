package com.spbsu.exp.dl.cuda;

import com.spbsu.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.spbsu.exp.dl.cuda.data.FMatrix;
import com.spbsu.exp.dl.cuda.data.FVector;
import com.spbsu.exp.dl.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.cuda.data.impl.FArrayVector;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 14.October.2014 at 12:23
 */
public class JcublasHelper {

  static {
    JcudaHelper.warmUp();
  }

  public static ColsVecArrayMx mult(final @NotNull ColsVecArrayMx A, final @NotNull ColsVecArrayMx B) {
    final int m = A.rows();
    final int k = A.columns();
    final int n = B.columns();
    final int mk = m * k;
    final int kn = k * n;
    final int mn = m * n;

    final double[] hA = A.toColumnMajor();
    final double[] hB = B.toColumnMajor();
    final double[] hC = new double[mn];

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer dB = new Pointer();
    final Pointer dC = new Pointer();

    JCublas.cublasAlloc(mk, Sizeof.DOUBLE, dA);
    JCublas.cublasAlloc(kn, Sizeof.DOUBLE, dB);
    JCublas.cublasAlloc(mn, Sizeof.DOUBLE, dC);

    JCublas.cublasSetVector(mk, Sizeof.DOUBLE, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(kn, Sizeof.DOUBLE, Pointer.to(hB), 1, dB, 1);
    JCublas.cublasSetVector(mn, Sizeof.DOUBLE, Pointer.to(hC), 1, dC, 1);

    JCublas.cublasDgemm('n', 'n', m, n, k, 1., dA, m, dB, k, 0., dC, m);

    JCublas.cublasGetVector(mn, Sizeof.DOUBLE, dC, 1, Pointer.to(hC), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(dB);
    JCublas.cublasFree(dC);

    JCublas.cublasShutdown();

    return new ColsVecArrayMx(n, hC);
  }

  public static FMatrix fMult(final @NotNull FMatrix A, final @NotNull FMatrix B) {
    final int rows = A.getRows();
    final float[] C = fMMmult(rows, A.getColumns(), B.getColumns(), A.toArray(), B.toArray());

    return new FArrayMatrix(rows, C);
  }

  public static float fDot(final @NotNull FVector a, final @NotNull FVector b) {
    return fDot(b.getDimension(), a.toArray(), b.toArray());
  }

  public static FVector fMult(final @NotNull FMatrix A, final @NotNull FVector b) {
    return new FArrayVector(fMVmult(A.getRows(), A.getColumns(), A.toArray(), false, b.toArray()));
  }

  public static FVector fMult(final @NotNull FVector b, final @NotNull FMatrix A) {
    return new FArrayVector(fMVmult(A.getRows(), A.getColumns(), A.toArray(), true, b.toArray()));
  }

  public static FMatrix fMult(final @NotNull FVector a, final @NotNull FVector b) {
    return new FArrayMatrix(a.getDimension(), fMMmult(a.getDimension(), 1, b.getDimension(), a.toArray(), b.toArray()));
  }

  public static FMatrix fSum(final @NotNull FMatrix A, final @NotNull FMatrix B) {
    return new FArrayMatrix(A.getRows(), fVVsum(A.toArray(), B.toArray(), 1.f));
  }

  public static FVector fSum(final @NotNull FVector a, final @NotNull FVector b) {
    return new FArrayVector(fVVsum(a.toArray(), b.toArray(), 1.f));
  }

  public static FMatrix fSubtr(final @NotNull FMatrix A, final @NotNull FMatrix B) {
    return new FArrayMatrix(A.getRows(), fVVsum(B.toArray(), A.toArray(), -1.f));
  }

  public static FVector fSubtr(final @NotNull FVector a, final @NotNull FVector b) {
    return new FArrayVector(fVVsum(a.toArray(), b.toArray(), -1.f));
  }

  public static FMatrix fScale(final @NotNull FMatrix A, final float alpha) {
    fVscale(A.toArray(), alpha);
    return A;
  }

  public static FVector fScale(final @NotNull FVector a, final float alpha) {
    fVscale(a.toArray(), alpha);
    return a;
  }

  private static float fDot(final int n, final float[] ha, final float[] hb) {
    JCublas.cublasInit();

    final Pointer da = new Pointer();
    final Pointer db = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.FLOAT, da);
    JCublas.cublasAlloc(n, Sizeof.FLOAT, db);

    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(ha), 1, da, 1);
    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(hb), 1, db, 1);

    final float hc = JCublas.cublasSdot(n, da, 1, db, 1);

    JCublas.cublasFree(da);
    JCublas.cublasFree(db);

    JCublas.cublasShutdown();

    return hc;
  }

  private static float[] fVVsum(final float[] ha, final float[] hb, final float alpha) {
    final int n = ha.length;
    final float[] hc = new float[n];

    JCublas.cublasInit();

    final Pointer da = new Pointer();
    final Pointer db = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.FLOAT, da);
    JCublas.cublasAlloc(n, Sizeof.FLOAT, db);

    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(ha), 1, da, 1);
    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(hb), 1, db, 1);

    JCublas.cublasSaxpy(n, alpha, da, 1, db, 1);

    JCublas.cublasGetVector(n, Sizeof.FLOAT, db, 1, Pointer.to(hc), 1);

    JCublas.cublasFree(da);
    JCublas.cublasFree(db);

    JCublas.cublasShutdown();

    return hc;
  }

  private static void fVscale(final float[] ha, final float alpha) {
    final int n = ha.length;

    JCublas.cublasInit();

    final Pointer da = new Pointer();

    JCublas.cublasAlloc(n, Sizeof.FLOAT, da);

    JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(ha), 1, da, 1);

    JCublas.cublasSscal(n, alpha, da, 1);

    JCublas.cublasGetVector(n, Sizeof.FLOAT, da, 1, Pointer.to(ha), 1);

    JCublas.cublasFree(da);

    JCublas.cublasShutdown();
  }

  private static float[] fMVmult(int m, int n, final float[] hA, final boolean trans, final float[] hb) {
    final int mn = m * n;
    final char op = trans ? 't' : 'n';
    final float[] hc = new float[trans ? n : m];

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer db = new Pointer();
    final Pointer dc = new Pointer();

    JCublas.cublasAlloc(mn, Sizeof.FLOAT, dA);
    JCublas.cublasAlloc(trans ? m : n, Sizeof.FLOAT, db);
    JCublas.cublasAlloc(trans ? n : m, Sizeof.FLOAT, dc);

    JCublas.cublasSetVector(mn, Sizeof.FLOAT, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(trans ? m : n, Sizeof.FLOAT, Pointer.to(hb), 1, db, 1);
    JCublas.cublasSetVector(trans ? n : m, Sizeof.FLOAT, Pointer.to(hc), 1, dc, 1);

    JCublas.cublasSgemv(op, m, n, 1.f, dA, m, db, 1, 0.f, dc, 1);

    JCublas.cublasGetVector(trans ? n : m, Sizeof.FLOAT, dc, 1, Pointer.to(hc), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(db);
    JCublas.cublasFree(dc);

    JCublas.cublasShutdown();

    return hc;
  }

  private static float[] fMMmult(final int m, final int k, final int n, final float[] hA, final float[] hB) {
    final int mk = m * k;
    final int kn = k * n;
    final int mn = m * n;

    final float[] hC = new float[mn];

    JCublas.cublasInit();

    final Pointer dA = new Pointer();
    final Pointer dB = new Pointer();
    final Pointer dC = new Pointer();

    JCublas.cublasAlloc(mk, Sizeof.FLOAT, dA);
    JCublas.cublasAlloc(kn, Sizeof.FLOAT, dB);
    JCublas.cublasAlloc(mn, Sizeof.FLOAT, dC);

    JCublas.cublasSetVector(mk, Sizeof.FLOAT, Pointer.to(hA), 1, dA, 1);
    JCublas.cublasSetVector(kn, Sizeof.FLOAT, Pointer.to(hB), 1, dB, 1);
    JCublas.cublasSetVector(mn, Sizeof.FLOAT, Pointer.to(hC), 1, dC, 1);

    JCublas.cublasSgemm('n', 'n', m, n, k, 1.f, dA, m, dB, k, 0.f, dC, m);

    JCublas.cublasGetVector(mn, Sizeof.FLOAT, dC, 1, Pointer.to(hC), 1);

    JCublas.cublasFree(dA);
    JCublas.cublasFree(dB);
    JCublas.cublasFree(dC);

    JCublas.cublasShutdown();

    return hC;
  }

  private static void swap(int a, int b) {
    a = a - b;
    b = a + b;
    a = b - a;
  }

}
