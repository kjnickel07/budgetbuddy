"""Open the model.tar.gz that SageMaker produced and load the ModernBERT (ONNX) model inside it."""
import json
import os
import tarfile

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OutputLengthModel:
    """Reads a prompt, predicts how many output tokens Llama will write."""

    def __init__(self, folder):
        with open(os.path.join(folder, "bb_config.json")) as f:
            config = json.load(f)
        # ModernBERT's own tokenizer. Not the Llama one in app.py, which only counts input tokens.
        self.tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        self.tokenizer.enable_truncation(config["max_len"])  # same cut-off the model was trained with
        self.tokenizer.no_padding()
        self.session = ort.InferenceSession(os.path.join(folder, "model.onnx"), providers=["CPUExecutionProvider"])

    def predict(self, prompt):
        ids = np.array([self.tokenizer.encode(prompt).ids], dtype=np.int64)
        log_tokens = self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
        return float(np.expm1(log_tokens.reshape(-1)[0]))  # the model learned log(1 + tokens)


def load_model_from_tar(tar, extract_dir):
    """`tar` is a path to model.tar.gz, or an open stream of it (the Lambda streams it straight from S3)."""
    archive = tarfile.open(tar) if isinstance(tar, str) else tarfile.open(fileobj=tar, mode="r|gz")
    with archive:
        archive.extractall(extract_dir, filter="data")
    return OutputLengthModel(extract_dir)
