#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

VOCAB_SIZE=8000

RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/aze_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=data/ted_binarized/aze_spm"$VOCAB_SIZE"/

FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py

LANGS=(aze)

for i in ${!LANGS[*]}; do
  LANG=${LANGS[$i]}
  mkdir -p "$PROC_DDIR"/"$LANG"_eng
  for f in "$RAW_DDIR"/"$LANG"_eng/*.orig.*-eng  ; do
    src=`echo $f | sed 's/-eng$//g'`
    trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`
    if [ ! -f "$src" ]; then
      echo "src=$src, trg=$trg"
      python cut_corpus.py 0 < $f > $src
      python cut_corpus.py 1 < $f > $trg
    fi
  done

  # --- learn BPE with sentencepiece ---
  TRAIN_FILES="$RAW_DDIR"/"$LANG"_eng/ted-train.orig."$LANG","$RAW_DDIR"/"$LANG"_eng/ted-train.orig.eng
  echo "learning joint BPE over ${TRAIN_FILES}..."
  python "$SPM_TRAIN" \
	    --input=$TRAIN_FILES \
	    --model_prefix="$PROC_DDIR"/"$LANG"_eng/spm"$VOCAB_SIZE" \
	    --vocab_size=$VOCAB_SIZE \
	    --character_coverage=1.0 \
	    --model_type=bpe
  spm_model="$PROC_DDIR"/"$LANG"_eng/spm"$VOCAB_SIZE".model

  python "$SPM_ENCODE" \
	  --model=$spm_model \
	  --output_format=piece \
	  --inputs "$RAW_DDIR"/"$LANG"_eng/ted-train.orig."$LANG" "$RAW_DDIR"/"$LANG"_eng/ted-train.orig.eng  \
	  --outputs "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".eng \
	  --min-len 1 --max-len 200 
 
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
    python "$SPM_ENCODE" \
	    --model=$spm_model \
	    --output_format=piece \
	    --inputs "$RAW_DDIR"/"$LANG"_eng/ted-"$split".orig."$LANG" "$RAW_DDIR"/"$LANG"_eng/ted-"$split".orig.eng  \
	    --outputs "$PROC_DDIR"/"$LANG"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_eng/ted-"$split".spm"$VOCAB_SIZE".eng  
  done

  # -- fairseq binarization ---
  echo "Binarize the data..."
  fairseq-preprocess --source-lang $LANG --target-lang eng \
	  --joined-dictionary \
	  --trainpref "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE" \
	  --validpref "$PROC_DDIR"/"$LANG"_eng/ted-dev.spm"$VOCAB_SIZE" \
	  --testpref "$PROC_DDIR"/"$LANG"_eng/ted-test.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/"$LANG"_eng/

  echo "Binarize the data..."
  fairseq-preprocess --source-lang eng --target-lang $LANG \
	  --joined-dictionary \
	  --trainpref "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE" \
	  --validpref "$PROC_DDIR"/"$LANG"_eng/ted-dev.spm"$VOCAB_SIZE" \
	  --testpref "$PROC_DDIR"/"$LANG"_eng/ted-test.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/eng_"$LANG"/
done