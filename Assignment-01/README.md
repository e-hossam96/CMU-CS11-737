# Assignment-01 (Multilingual POS Tagging)

## Reproduce results

| en | cs | es | ar | af | lt | hy | ta |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 91.44 | 93.97 | 93.36 | 94.44 | 88.85 | 75.59 | 79.91 | 40.09 |

## CRF extension

### Setup
We make use of the module [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable/).
* Add the `CRF` layer in the model `__init__` method.
* Apply fixes to the `train` and `evaluate` functions to accommodate new predictions shape.
* Fix the `categorical_accuracy` function to work with the crf decoded predictions.

### Results
| en | cs | es | ar | af | lt | hy | ta |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 91.55 | 94.02 | 93.56 | 94.36 | 89.11 | 75.42 | 80.02 | 40.19 |


## GloVe Pretrained Embeddings

### Setup
* Load the embeddings usign `torchtext.vocab.GloVe`.
* Change embedding size in model config to 300.
* Initialize the model embeddings using `glove.get_vecs_by_tokens(vocab_text.vocab)`

### Results
| en | cs | es | ar | af | lt | hy | ta |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 91.48 | 93.74 | 93.22 | 94.56 | 91.49 | 76.96 | 81.11 | 47.64 |
