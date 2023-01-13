#!/bin/bash

set -euo pipefail

if [[ -z $FAIRSEQ_DIR ]]; then
  echo "\$FAIRSEQ_DIR enviromental variable needs to be set"
  exit 1
fi

VOCAB_SIZE=8000

RAW_DDIR=data/ted_raw/
PROC_DDIR=data/ted_processed/aztr_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=data/ted_binarized/aztr_spm"$VOCAB_SIZE"/

FAIR_SCRIPTS=$FAIRSEQ_DIR/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py

LANGS=(az tr)

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
done

# learn BPE with sentencepiece for English
TRAIN_FILES=$(for LANG in "${LANGS[@]}"; do echo "$RAW_DDIR"/"$LANG"_en/train.en; done | tr "\n" ",")
echo "learning BPE for en over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
      --input=$TRAIN_FILES \
      --model_prefix="$PROC_DDIR"/spm"$VOCAB_SIZE".en \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=1.0 \
      --model_type=bpe

# train a separate BPE model for each language, then encode the data with the corresponding BPE model
for LANG in ${LANGS[@]}; do
  TRAIN_FILES="$RAW_DDIR"/"$LANG"_en/train."$LANG"
  echo "learning BPE for ${TRAIN_FILES} ..."
  python "$SPM_TRAIN" \
        --input=$TRAIN_FILES \
        --model_prefix="$PROC_DDIR"/"$LANG"_en/spm"$VOCAB_SIZE" \
        --vocab_size=$VOCAB_SIZE \
        --character_coverage=1.0 \
        --model_type=bpe

  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".en.model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_en/train.en  \
        --outputs "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE".prefilt.en 
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/"$LANG"_en/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_en/train."$LANG"  \
        --outputs "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE".prefilt."$LANG" 
  python clean_corpus.py \
    --inputs "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE".prefilt."$LANG" "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE".prefilt.en \
    --outputs "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE"."$LANG" "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE".en \
    --min-len 1 --max-len 200 
 
  echo "encoding valid/test data with learned BPE..."
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/spm"$VOCAB_SIZE".en.model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_en/"$split".en  \
        --outputs "$PROC_DDIR"/"$LANG"_en/"$split".spm"$VOCAB_SIZE".en  
  done
  for split in dev test;
  do
  python "$SPM_ENCODE" \
        --model="$PROC_DDIR"/"$LANG"_en/spm"$VOCAB_SIZE".model \
        --output_format=piece \
        --inputs "$RAW_DDIR"/"$LANG"_en/"$split"."$LANG"  \
        --outputs "$PROC_DDIR"/"$LANG"_en/"$split".spm"$VOCAB_SIZE"."$LANG" 
  done
done


# Concatenate all the training data from all languages to get combined vocabulary
mkdir -p $BINARIZED_DDIR
mkdir -p $BINARIZED_DDIR/M2O/
mkdir -p $BINARIZED_DDIR/O2M/

for LANG in ${LANGS[@]}; do
  cat $PROC_DDIR/"$LANG"_en/train.spm"$VOCAB_SIZE"."$LANG" >> $BINARIZED_DDIR/combined-train.src
  cat $PROC_DDIR/"$LANG"_en/train.spm"$VOCAB_SIZE".en >> $BINARIZED_DDIR/combined-train.en
done
fairseq-preprocess -s src -t en \
  --trainpref $BINARIZED_DDIR/combined-train \
  --joined-dictionary \
  --workers 8 \
  --thresholdsrc 0 \
  --thresholdtgt 0 \
  --destdir $BINARIZED_DDIR

echo "Binarize the data..."
for LANG in ${LANGS[@]}; do
  # Binarize the data for many-to-one translation
  fairseq-preprocess --source-lang $LANG --target-lang en \
        --trainpref "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE" \
        --validpref "$PROC_DDIR"/"$LANG"_en/dev.spm"$VOCAB_SIZE" \
        --testpref "$PROC_DDIR"/"$LANG"_en/test.spm"$VOCAB_SIZE" \
      	--srcdict $BINARIZED_DDIR/dict.src.txt \
      	--tgtdict $BINARIZED_DDIR/dict.en.txt \
        --destdir $BINARIZED_DDIR/M2O/
  
  # Binarize the data for one-to-many translation
  fairseq-preprocess --source-lang en --target-lang $LANG \
        --trainpref "$PROC_DDIR"/"$LANG"_en/train.spm"$VOCAB_SIZE" \
        --validpref "$PROC_DDIR"/"$LANG"_en/dev.spm"$VOCAB_SIZE" \
        --testpref "$PROC_DDIR"/"$LANG"_en/test.spm"$VOCAB_SIZE" \
      	--srcdict $BINARIZED_DDIR/dict.en.txt \
      	--tgtdict $BINARIZED_DDIR/dict.src.txt \
        --destdir $BINARIZED_DDIR/O2M/
done
