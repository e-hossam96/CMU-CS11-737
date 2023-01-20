# Assignment-02 (Multilingual Translation)

This assignment requires a lot of work to debug the codes and get the files to run.

## Debugging

### Environment

* Follow the steps in the assignment [webpage](http://phontron.com/class/multiling2022/assignment2.html). Make sure to have `numpy` version less than `1.24`.
* Install [COMET](https://github.com/Unbabel/COMET) and [Fairseq](https://github.com/facebookresearch/fairseq) *v0.10.2* correctly and **export** its directory. This is an important step.

### Data

* The file [download_data.py](./download_data.py) **will not** work correctly so skip it.
* Run the command `bash get_data.sh` in the terminal and the data from the main [repo](https://github.com/neulab/word-embeddings-for-nmt) and preprocessing file will be downloaded.
* Change the `src_lang` and `trg_lang` for each pair in the experiment as follows: **low resource** (`src_lang`) and **high resource** (`trg_lang`).

### Bash Files

Once the data is provided, we need to fix all bash file to work properly.
* Change languages names to follow the ISO 639-1 coding in codes and file naming.
* Apply fixes to the source and target languages paths. Specifically, remove all `.orig` and `ted-` from all files.
* Make sure to provide the correct directory for the `COMET` module.
* Now, we can run each experiment correcly without errors.


## Reproducing Results

### Bilingual Baselines

Once we run the experiment. The following results are produced.

| Pair | az - en | en - az | be - en | en - be |
| :--: | :--: | :--: | :--: | :--: |
| BLUE | 3.01 | 20.21 | 4.67 | 11.46 |
| COMET | -1.4159 | -1.2864 | -1.3589 | -1.3506 |

As seen from the results, when changing to `en` being the source language we get much higher scores. This is to be investigated.

Note: The reported results are the test sets'. For extended results see [reproduce_results](./reproduce_results.txt)

### Multilingual Training

First of all run `pip install --upgrade lxml`.

| Pair | az - en | en - az | be - en | en - be |
| :--: | :--: | :--: | :--: | :--: |
| BLUE | 14.31 | 5.92 | 18.90 | 9.40 |
| COMET | -0.2684 | -0.0691 | -0.3816 | -0.4756 |

Note: `Az` is enriched with `Tr` and `Be` is enriched with `Ru`.


### Finetuning Pretrained Multilingual Models

| Pair | az - en | en - az | be - en | en - be |
| :--: | :--: | :--: | :--: | :--: |
| BLUE | - | - | - | - |
| COMET | - | - | - | - |