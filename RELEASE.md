# Release v1.1.0

## Breaking changes

* This only affect you if you use `BertTokenizer` or `BertEncoder` in AutoKeras
  explicity.  You are not affected if you only use `BertBlock`, `TextClassifier`
  or `TextRegressor`.  Removed the AutoKeras implementation of `BertTokenizer`
  and `BertEncoder`.  Use `keras-nlp` implementation instead.

## New features

## Bug fixes
