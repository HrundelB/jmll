package com.spbsu.exp.cuda.process.functions;

import com.spbsu.exp.cuda.process.functions.floats.HeavisideFA;
import com.spbsu.exp.cuda.process.functions.floats.IdenticalFA;
import com.spbsu.exp.cuda.process.functions.floats.SigmoidFA;
import com.spbsu.exp.cuda.process.functions.floats.TanhFA;

/**
 * jmll
 * ksen
 * 13.December.2014 at 14:01
 */
public enum FunctionType {

  HEAVISIDE(){
    HeavisideFA getInstance() {
      return new HeavisideFA();
    }
  },
  IDENTICAL(){
    IdenticalFA getInstance() {
      return new IdenticalFA();
    }
  },
  SIGMOID(){
    SigmoidFA getInstance() {
      return new SigmoidFA();
    }
  },
  TANH(){
    TanhFA getInstance() {
      return new TanhFA();
    }
  }
  ;

  abstract ArrayUnaryFunction getInstance();

}
