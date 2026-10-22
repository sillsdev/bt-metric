"""Scores batches with a COMET model, reading requests from stdin and writing results to stdout as JSON lines.

Each request is {"src": [...], "mt": [...], "ref": [...]} ("ref" omitted for quality estimation models) and each
response is {"scores": [...]} or {"error": "..."}. The model is loaded once, so many evaluation sets can be scored
without paying the load time again.
"""

import argparse
import contextlib
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="A COMET model name, e.g. Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import logging
    import warnings

    import torch
    from comet import download_model, load_from_checkpoint

    warnings.filterwarnings("ignore")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    # TF32 matmuls are much faster on recent GPUs; COMET scores are unaffected at the precision they're reported.
    torch.set_float32_matmul_precision("high")

    # COMET and Lightning print progress to stdout, which is reserved for responses.
    with contextlib.redirect_stdout(sys.stderr):
        model = load_from_checkpoint(download_model(args.model))
    print(json.dumps({"ready": args.model}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            keys = [key for key in ("src", "mt", "ref") if key in request]
            data = [dict(zip(keys, values)) for values in zip(*(request[key] for key in keys))]
            with contextlib.redirect_stdout(sys.stderr):
                output = model.predict(data, batch_size=args.batch_size, gpus=1, progress_bar=False)
            response = {"scores": [float(score) for score in output.scores]}
        except Exception as e:  # noqa: BLE001 - any failure is reported back instead of killing the worker
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
