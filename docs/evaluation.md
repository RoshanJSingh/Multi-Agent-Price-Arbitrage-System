# Evaluation

The evaluation path is anchored on normalized Amazon Reviews 2023 product metadata. The target baseline size is 40,000 records, with each row converted into:

- product title and description text,
- category metadata,
- numeric ground-truth price,
- source identifier for reproducible indexing.

Model comparisons use MAE, RMSE, and median absolute percentage error. Deal quality is measured by absolute discount, calculated as estimated fair value minus live deal price.

