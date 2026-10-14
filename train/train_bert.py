"""SageMaker entry point: fine-tune ModernBERT to read a prompt and predict log(1 + output tokens).

Inputs  (S3 -> container): /opt/ml/input/data/{train,validation}/data.csv  with columns prompt_text,target
Outputs (container -> S3): /opt/ml/model/ -> SageMaker tars it into model.tar.gz
    model.onnx       the network, weights squeezed to 8 bits so it is small and fast on a Lambda CPU
    tokenizer.json   ModernBERT's tokenizer (NOT the Llama one; that only counts input tokens)
    bb_config.json   max_len + how to turn the network's number back into tokens
    metrics.json     validation MAE, so you can compare runs without downloading anything else
Also: /opt/ml/output/data/hf_model -> output.tar.gz, the plain PyTorch weights (backup, not used by the Lambda)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--model_name", default="answerdotai/ModernBERT-base")
p.add_argument("--max_len", type=int, default=1024)      # tokens of the prompt the model reads (ModernBERT allows 8192)
p.add_argument("--epochs", type=float, default=4)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--max_rows", type=int, default=0)        # >0 = quick smoke test on a few rows
p.add_argument("--train_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
p.add_argument("--val_dir", default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
p.add_argument("--work_dir", default="/tmp/hf")         # checkpoints; NOT shipped in model.tar.gz
p.add_argument("--output_data_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
args = p.parse_args()


def load(folder):
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    if args.max_rows:
        df = df.head(args.max_rows)
    df["prompt_text"] = df["prompt_text"].fillna("")
    df["labels"] = np.log1p(df["target"]).astype("float32")  # long tail: 2 to 12,570 tokens, so learn the log
    return df


train_df, val_df = load(args.train_dir), load(args.val_dir)
tok = AutoTokenizer.from_pretrained(args.model_name)


def to_dataset(df):
    enc = tok(df["prompt_text"].tolist(), truncation=True, max_length=args.max_len)
    return [{"input_ids": ids, "attention_mask": mask, "labels": label}
            for ids, mask, label in zip(enc["input_ids"], enc["attention_mask"], df["labels"])]


train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, num_labels=1, problem_type="regression", attn_implementation="sdpa")


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    return {"mae": float(np.abs(np.expm1(preds.reshape(-1)) - np.expm1(labels)).mean())}  # in output tokens


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=args.work_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=0.1,                        # a float = 10% of all training steps
        weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        train_sampling_strategy="group_by_length",  # batches prompts of similar length -> much less padding
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tok),
    processing_class=tok,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)
trainer.train()
torch_mae = trainer.evaluate()["eval_mae"]
print(f"validation MAE (PyTorch): {torch_mae:.1f} output tokens")

# Keep the PyTorch weights too, in SageMaker's separate output.tar.gz (the Lambda never downloads it).
# If the ONNX export below ever fails, you can still export these on your laptop instead of retraining.
trainer.save_model(os.path.join(args.output_data_dir, "hf_model"))


# ---- Export: PyTorch -> ONNX -> 8-bit, so the Lambda needs only onnxruntime (no 2 GB torch install) ----
class Regressor(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


os.makedirs(args.model_dir, exist_ok=True)
fp32_path, onnx_path = os.path.join(args.work_dir, "model_fp32.onnx"), os.path.join(args.model_dir, "model.onnx")
net = Regressor(trainer.model.float().cpu()).eval()
example = tok(["export example"], return_tensors="pt")
batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq")
torch.onnx.export(
    net, (example["input_ids"], example["attention_mask"]), fp32_path,
    input_names=["input_ids", "attention_mask"], output_names=["log_tokens"],
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    dynamo=True, external_data=False,
)

import logging  # noqa: E402
import onnx  # noqa: E402
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer  # noqa: E402

# 8-bit weights with one scale per block of 32 numbers: about 600 MB -> 280 MB and ~1% off PyTorch.
# (Plain int8 "quantize_dynamic" uses one scale per whole matrix and was off by ~2x on ModernBERT.)
logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
quantizer = MatMulNBitsQuantizer(onnx.load(fp32_path), bits=8, block_size=32, is_symmetric=True)
quantizer.process()
quantizer.model.save_model_to_file(onnx_path, False)

tok.backend_tokenizer.save(os.path.join(args.model_dir, "tokenizer.json"))
with open(os.path.join(args.model_dir, "bb_config.json"), "w") as f:
    json.dump({"max_len": args.max_len, "target": "log1p", "base_model": args.model_name}, f)


# ---- Check: does the shipped 8-bit ONNX model still agree with PyTorch? ----
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
check = val_df.head(100)
torch_pred, onnx_pred = [], []
with torch.no_grad():
    for text in check["prompt_text"]:
        enc = tok(text, truncation=True, max_length=args.max_len, return_tensors="np")
        feeds = {"input_ids": enc["input_ids"].astype("int64"), "attention_mask": enc["attention_mask"].astype("int64")}
        onnx_pred.append(float(sess.run(None, feeds)[0].reshape(-1)[0]))
        torch_pred.append(float(net(torch.from_numpy(feeds["input_ids"]), torch.from_numpy(feeds["attention_mask"])).reshape(-1)[0]))
torch_tokens, onnx_tokens = np.expm1(torch_pred), np.expm1(onnx_pred)
rel_diff = float(np.median(np.abs(onnx_tokens - torch_tokens) / np.maximum(torch_tokens, 1)))
onnx_mae = float(np.abs(onnx_tokens - check["target"]).mean())
print(f"8-bit ONNX vs PyTorch on {len(check)} validation prompts: median difference {rel_diff:.1%}")

metrics = {"val_mae_torch": torch_mae, "onnx_vs_torch_median_rel_diff": rel_diff,
           "val_mae_onnx_first100": onnx_mae, "val_mae_torch_first100": float(np.abs(torch_tokens - check["target"]).mean())}
with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1)
print(json.dumps(metrics, indent=1))

del sess  # free onnxruntime threads before Python shuts down (otherwise macOS can abort at exit)
