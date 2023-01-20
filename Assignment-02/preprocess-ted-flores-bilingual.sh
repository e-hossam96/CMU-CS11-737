#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

FLORES_MODEL=checkpoints/flores101_mm100_175M
SPM_MODEL=$FLORES_MODEL/sentencepiece.bpe.model
DICT_FILE=$FLORES_MODEL/dict.txt

RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/be_spm_flores/
BINARIZED_DDIR=data/ted_binarized/be_flores/

FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py

LANGS=(be)

for i in ${!LANGS[*]}; do
  LANG=${LANGS[$i]}
  mkdir -p "$PROC_DDIR"/"$LANG"_en
  for f in "$RAW_DDIR"/"$LANG"_en/*.en  ; do
    src=`echo $f | sed 's/-en$//g'`
    trg=`echo $f | sed 's/\.[^\.]*$/.en/g'`
    if [ ! -f "$src" ]; then
      echo "src=$src, trg=$trg"
      python cut_corpus.py 0 < $f > $src
      python cut_corpus.py 1 < $f > $trg
    fi
  done

  LANG_CODE=${LANG:0:2}
 
  # --- apply BPE with sentencepiece ---
  python "$SPM_ENCODE" \
	  --model=$SPM_MODEL \
	  --output_format=piece \
	  --inputs "$RAW_DDIR"/"$LANG"_en/train."$LANG" "$RAW_DDIR"/"$LANG"_en/train.en  \
	  --outputs "$PROC_DDIR"/"$LANG"_en/train.spm."$LANG_CODE" "$PROC_DDIR"/"$LANG"_en/train.spm.en \
    --min-len 1 --max-len 200 

  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
    python "$SPM_ENCODE" \
	    --model=$SPM_MODEL \
	    --output_format=piece \
	    --inputs "$RAW_DDIR"/"$LANG"_en/"$split"."$LANG" "$RAW_DDIR"/"$LANG"_en/"$split".en  \
	    --outputs "$PROC_DDIR"/"$LANG"_en/"$split".spm."$LANG_CODE" "$PROC_DDIR"/"$LANG"_en/"$split".spm.en  
  done

  # -- fairseq binarization ---
  echo "Binarize the data... (be-en)"
  fairseq-preprocess --source-lang $LANG_CODE --target-lang en \
	  --srcdict $DICT_FILE --tgtdict $DICT_FILE --thresholdsrc 0 --thresholdtgt 0 \
	  --trainpref "$PROC_DDIR"/"$LANG"_en/train.spm \
	  --validpref "$PROC_DDIR"/"$LANG"_en/dev.spm \
	  --testpref "$PROC_DDIR"/"$LANG"_en/test.spm \
	  --destdir $BINARIZED_DDIR/"$LANG"_en/

  echo "Binarize the data... (en-be)"
  fairseq-preprocess --source-lang en --target-lang $LANG_CODE \
	  --srcdict $DICT_FILE --tgtdict $DICT_FILE --thresholdsrc 0 --thresholdtgt 0 \
	  --trainpref "$PROC_DDIR"/"$LANG"_en/train.spm \
	  --validpref "$PROC_DDIR"/"$LANG"_en/dev.spm \
	  --testpref "$PROC_DDIR"/"$LANG"_en/test.spm \
	  --destdir $BINARIZED_DDIR/en_"$LANG"/
done