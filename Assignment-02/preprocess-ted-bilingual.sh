#!/bin/bash

set -euo pipefail


if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

VOCAB_SIZE=8000

RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/be_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=data/ted_binarized/be_spm"$VOCAB_SIZE"/

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

  # --- learn BPE with sentencepiece ---
  TRAIN_FILES="$RAW_DDIR"/"$LANG"_en/train."$LANG","$RAW_DDIR"/"$LANG"_en/train.en
  echo "learning joint BPE over ${TRAIN_FILES}..."
  python "$SPM_TRAIN" \
	    --input=$TRAIN_FILES \
	    --model_prefix="$PROC_DDIR"/"$LANG"_en/spm"$VOCAB_SIZE" \
	    --vocab_size=$VOCAB_SIZE \
	    --character_coverage=1.0 \
	    --model_type=bpe
  spm_model="$PROC_DDIR"/"$LANG"_en/spm"$VOCAB_SIZE".model

  python "$SPM_ENCODE" \
	  --model=$spm_model \
	  --output_format=piece \
	  --inputs "$RAW_DDIR"/"$LANG"_en/train."$LANG" "$RAW_DDIR"/"$LANG"_en/train.en  \
	  --outputs "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE".en \
	  --min-len 1 --max-len 200 
 
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
    python "$SPM_ENCODE" \
	    --model=$spm_model \
	    --output_format=piece \
	    --inputs "$RAW_DDIR"/"$LANG"_en/"$split"."$LANG" "$RAW_DDIR"/"$LANG"_en/"$split".en  \
	    --outputs "$PROC_DDIR"/"$LANG"_en/"$split".spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_en/"$split".spm"$VOCAB_SIZE".en  
  done

  # -- fairseq binarization ---
  echo "Binarize the data..."
  fairseq-preprocess --source-lang $LANG --target-lang en \
	  --joined-dictionary \
	  --trainpref "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE" \
	  --validpref "$PROC_DDIR"/"$LANG"_en/dev.spm"$VOCAB_SIZE" \
	  --testpref "$PROC_DDIR"/"$LANG"_en/test.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/"$LANG"_en/

  echo "Binarize the data..."
  fairseq-preprocess --source-lang en --target-lang $LANG \
	  --joined-dictionary \
	  --trainpref "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE" \
	  --validpref "$PROC_DDIR"/"$LANG"_en/dev.spm"$VOCAB_SIZE" \
	  --testpref "$PROC_DDIR"/"$LANG"_en/test.spm"$VOCAB_SIZE" \
	  --destdir $BINARIZED_DDIR/en_"$LANG"/
done