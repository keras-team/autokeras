# Release v2.0.0

## Breaking changes

* Requires `keras>=3.0.0` instead of `tf.keras`.
* Removed the structured data related tasks by removing the following public
  APIs:
  * `CategoricalToNumerical`
  * `MultiCategoryEncoding`
  * `StructuredDataInput`
  * `StructuredDataBlock`
  * `StructuredDataClassifier`
  * `StructuredDataRegressor`
* Removed the Time series related tasks by removing the following public APIs:
  * `TimeseriesInput`
  * `TimeseriesForecaster`
* Reduced search space of Text related tasks by removing the following blocks.
  * `Embedding`
  * `TextToIntSequence`
  * `TextToNgramVector`
  * `Transformer`

# Release v1.1.0

## Breaking changes

* This only affect you if you use `BertTokenizer` or `BertEncoder` in AutoKeras
  explicity.  You are not affected if you only use `BertBlock`, `TextClassifier`
  or `TextRegressor`.  Removed the AutoKeras implementation of `BertTokenizer`
  and `BertEncoder`.  Use `keras-nlp` implementation instead.

## New features

## Bug fixes
* Now also support `numpy>=1.24`.
