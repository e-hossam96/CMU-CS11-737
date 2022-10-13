# multilingual-nlp-assignment-2

### download data:
First, you can download the data using
```python
download_data.py
``` 

### fairseq:
The preprocess and experiment scripts work with fairseq. To use fairseq, first clone and install it from [here](https://github.com/pytorch/fairseq/).
You might need to install two more python packages. Simply do:
```
pip install importlib_metadata
```

```
pip install sacremoses
```

### preprocess data:
To preprocess the data for bilingual training, please do
```bash
preprocess_scripts/make-ted-bilingual.sh
```

To preprocess the data for multilingual training, please do
```base
preprocess_scripts/make-ted-multilingual.sh
```

### training and translation scripts:
To submit training and translation experiment for bilingual setting for aze-eng, use the script
```bash
job_scripts/aze_eng_spm8000.sh
```

To submit training and translation experiment for multilingual setting for aze-eng, use the script
```bash
job_scripts/azetur_m2o_sepspm8000.sh
```

To submit training and translation experiment for bilingual setting for eng-aze, use the script
```bash
job_scripts/eng_aze_spm8000.sh
```

To submit training and translation experiment for multilingual setting for eng-aze, use the script
```bash
job_scripts/azetur_o2m_sepspm8000.sh
```
If you are using GPUs, remember to configure the CUDA_VISIBLE_DECIVE to different number if you are running muliple experiments at the same time.
