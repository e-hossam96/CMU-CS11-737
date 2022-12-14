import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cleans a set of parallel corpus")
    parser.add_argument("--inputs", type=str, nargs="+")
    parser.add_argument("--outputs", type=str, nargs="+")
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=200)
    args = parser.parse_args()
    input_corpora = zip(
        *[
            [l.strip() for l in open(f, encoding="utf-8").readlines()]
            for f in args.inputs
        ]
    )
    output_files = [
        open(f, "w", encoding="utf-8") for f in args.outputs
    ]
    for lines in input_corpora:
        include = True
        for line in lines:
            size = len(line.split(" "))
            if size < args.min_len or size > args.max_len:
                include = False
                break
        if include:
            for i, line in enumerate(lines):
                print(line, file=output_files[i])
