#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

VOCAB_SIZE=8000

RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/azetur_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=data/ted_binarized/azetur_spm"$VOCAB_SIZE"/

FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py

LANGS=(aze tur)

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
done

# learn BPE with sentencepiece for English
TRAIN_FILES=$(for LANG in "${LANGS[@]}"; do echo "$RAW_DDIR"/"$LANG"_eng/ted-train.orig.eng; done | tr "\n" ",")
echo "learning BPE for eng over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
      --input=$TRAIN_FILES \
      --model_prefix="$PROC_DDIR"/spm"$VOCAB_SIZE".eng \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=1.0 \
      --model_type=bpe

# train a separate BPE model for each language, then encode the data with the corresponding BPE model
for LANG in ${LANGS[@]}; do
  TRAIN_FILES="$RAW_DDIR"/"$LANG"_eng/ted-train.orig."$LANG"
  echo "learning BPE for ${TRAIN_FILES} ..."
  python "$SPM_TRAIN" \
        --input=$TRAIN_FILES \
        --model_prefix="$PROC_DDIR"/"$LANG"_eng/spm"$VOCAB_SIZE" \
        --vocab_size=$VOCAB_SIZE \
        --character_coverage=1.0 \
        --model_type=bpe

  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".eng.model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_eng/ted-train.orig.eng  \
        --outputs "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".prefilt.eng 
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/"$LANG"_eng/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_eng/ted-train.orig."$LANG"  \
        --outputs "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".prefilt."$LANG" 
  python clean_corpus.py \
    --inputs "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".prefilt."$LANG" "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".prefilt.eng \
    --outputs "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".eng \
    --min-len 1 --max-len 200 
 
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".eng.model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_eng/ted-"$split".orig.eng  \
        --outputs "$PROC_DDIR"/"$LANG"_eng/ted-"$split".spm"$VOCAB_SIZE".eng  
  done
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/"$LANG"_eng/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_eng/ted-"$split".orig."$LANG"  \
        --outputs "$PROC_DDIR"/"$LANG"_eng/ted-"$split".spm"$VOCAB_SIZE"."$LANG" 
  done
done


# Concatenate all the training data from all languages to get combined vocabulary
mkdir -p $BINARIZED_DDIR
mkdir -p $BINARIZED_DDIR/M2O/
mkdir -p $BINARIZED_DDIR/O2M/

for LANG in ${LANGS[@]}; do
  cat $PROC_DDIR/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE"."$LANG" >> $BINARIZED_DDIR/combined-train.src
  cat $PROC_DDIR/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE".eng >> $BINARIZED_DDIR/combined-train.eng
done
fairseq-preprocess -s src -t eng \
  --trainpref $BINARIZED_DDIR/combined-train \
  --joined-dictionary \
  --workers 8 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $BINARIZED_DDIR

echo "Binarize the data..."
for LANG in ${LANGS[@]}; do
  # Binarize the data for many-to-one translation
  fairseq-preprocess --source-lang $LANG --target-lang eng \
        --trainpref "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE" \
        --validpref "$PROC_DDIR"/"$LANG"_eng/ted-dev.spm"$VOCAB_SIZE" \
        --testpref "$PROC_DDIR"/"$LANG"_eng/ted-test.spm"$VOCAB_SIZE" \
      	--srcdict $BINARIZED_DDIR/dict.src.txt \
      	--tgtdict $BINARIZED_DDIR/dict.eng.txt \
        --destdir $BINARIZED_DDIR/M2O/
  
  # Binarize the data for one-to-many translation
  fairseq-preprocess --source-lang eng --target-lang $LANG \
        --trainpref "$PROC_DDIR"/"$LANG"_eng/ted-train.spm"$VOCAB_SIZE" \
        --validpref "$PROC_DDIR"/"$LANG"_eng/ted-dev.spm"$VOCAB_SIZE" \
        --testpref "$PROC_DDIR"/"$LANG"_eng/ted-test.spm"$VOCAB_SIZE" \
      	--srcdict $BINARIZED_DDIR/dict.eng.txt \
      	--tgtdict $BINARIZED_DDIR/dict.src.txt \
        --destdir $BINARIZED_DDIR/O2M/
done
