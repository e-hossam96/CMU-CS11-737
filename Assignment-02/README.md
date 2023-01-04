# Assignment-02 (Multilingual Translation)

This assignment requires a lot of work to debug the codes and get the files to run.

## Debugging

### Environment
Follow the steps in the assignment [webpage](http://phontron.com/class/multiling2022/assignment2.html).

### Data
* The file [download_data.py](./download_data.py) **will not** work correctly so skip it.
* Run the command `bash get_data.sh` in the terminal and the data from the main [repo](https://github.com/neulab/word-embeddings-for-nmt) and preprocessing file will be downloaded.
* Change the src_lang and trg_lang for each pair in the experiment as follows: low resource (src_lagn) and high resource (trg_lang).

### Bash Files
Once the data is provided, we need to fix all bash file to work properly.
* Change languages names to follow the ISO 639-1 coding in codes and file naming.
* Apply fixes to the source and target languages paths. Specifically, remove all `.orig` and `ted-` from all files.
* Now, we can run each experiment correcly without errors.


## Reproduce Results
To be continued ...
