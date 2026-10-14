"""The Lambda. API Gateway hands it a prompt, it hands back input + predicted output tokens."""
import json
import os

import boto3
from tokenizers import Tokenizer

from model_io import load_model_from_tar

MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ.get("MODEL_KEY", "models/current/model.tar.gz")

_model = None  # kept in memory between requests while the Lambda stays warm
_tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "tokenizer.json"))
CHAT_TEMPLATE_TOKENS = 35  # fixed wrapper around every prompt in the dataset; makes the count equal input_size


def _get_model():
    global _model
    if _model is None:
        # Unpack straight from S3 into /tmp (the only folder a Lambda may write to), without saving the
        # ~260 MB tar first, so /tmp only needs room for the unpacked model.
        stream = boto3.client("s3").get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)["Body"]
        _model = load_model_from_tar(stream, "/tmp/model")
    return _model


def _response(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def handler(event, context):
    try:
        prompt = json.loads(event.get("body") or "{}").get("prompt")
    except json.JSONDecodeError:
        return _response(400, {"error": "Body must be JSON"})
    if not isinstance(prompt, str) or not prompt:
        return _response(400, {"error": 'Send JSON like {"prompt": "your text"}'})

    output_tokens = max(0, round(_get_model().predict(prompt)))  # the model's guess, from reading the prompt
    input_tokens = len(_tokenizer.encode(prompt.strip()).ids) + CHAT_TEMPLATE_TOKENS  # counted, not guessed
    return _response(200, {
        "input_tokens": input_tokens,
        "predicted_output_tokens": output_tokens,
        "predicted_total_tokens": input_tokens + output_tokens,
    })
