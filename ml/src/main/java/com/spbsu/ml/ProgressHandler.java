package com.spbsu.ml;

import com.spbsu.commons.func.Action;

/**
 * User: solar
 * Date: 22.12.2010
 * Time: 17:17:41
 */
public interface ProgressHandler extends Action<Trans> {
  void invoke(Trans partial);
}