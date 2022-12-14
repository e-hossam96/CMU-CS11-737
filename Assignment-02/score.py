import argparse
import numpy as np
import sacrebleu

COMET_MODEL = "wmt20-comet-da"
COMET_BATCH_SIZE = 64
BLEURT_BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--comet-dir", type=str, default=None)
    parser.add_argument("--src", type=str)

    args = parser.parse_args()

    with open(args.hyp, encoding="utf-8") as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref, encoding="utf-8") as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]


    # gets corpus-level non-ml evaluation metrics
    # corpus-level BLEU
    bleu = sacrebleu.metrics.BLEU()
    print(bleu.corpus_score(hyps, [refs]).format())

    if args.comet_dir is not None:
        from comet import download_model, load_from_checkpoint

        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMET_MODEL, args.comet_dir)
        comet_model = load_from_checkpoint(comet_path)

        print("Running COMET evaluation...")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        # sentence-level and corpus-level COMET
        comet_sentscores, comet_score = comet_model.predict(
            comet_input, batch_size=COMET_BATCH_SIZE, sort_by_mtlen=True
        )

        print(f"COMET = {comet_score:.4f}")


if __name__ == "__main__":
    main()
