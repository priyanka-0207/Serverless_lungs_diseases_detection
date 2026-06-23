"""
Reference AWS Lambda handler for serverless chest X-ray inference.

Loads an ONNX EfficientNet-B2 model from S3, runs inference on a base64
image sent through API Gateway, generates an occlusion saliency heatmap,
writes the heatmap to S3 and the prediction to DynamoDB, and returns the
result as JSON.

This is a cleaned reference version. Resource names and the bucket/table
come from environment variables.
"""

import os
import json
import base64
import io
import uuid
import time

import boto3
import numpy as np
import onnxruntime as ort
from PIL import Image

S3_BUCKET = os.environ["S3_BUCKET"]
DYNAMO_TABLE = os.environ["DYNAMO_TABLE"]
MODEL_KEY = os.environ.get("MODEL_KEY", "model.onnx")

CLASSES = ["Normal", "Pneumonia", "ILD"]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

s3 = boto3.client("s3")
dynamo = boto3.resource("dynamodb").Table(DYNAMO_TABLE)

# Load the model once per container (reused across warm invocations)
_local_model = "/tmp/model.onnx"
if not os.path.exists(_local_model):
    s3.download_file(S3_BUCKET, MODEL_KEY, _local_model)
session = ort.InferenceSession(_local_model, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None, :]  # NCHW
    return arr.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def occlusion_saliency(arr: np.ndarray, base_class: int, patch: int = 32, stride: int = 32) -> np.ndarray:
    """Slide a gray patch across the image; drop in confidence = importance."""
    _, _, h, w = arr.shape
    heat = np.zeros((h, w), dtype=np.float32)
    base = softmax(session.run(None, {INPUT_NAME: arr})[0][0])[base_class]
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            occ = arr.copy()
            occ[:, :, y:y + patch, x:x + patch] = 0.0
            score = softmax(session.run(None, {INPUT_NAME: occ})[0][0])[base_class]
            heat[y:y + patch, x:x + patch] = base - score
    heat = np.clip(heat, 0, None)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def handler(event, context):
    body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event
    image_b64 = body["image"]
    filename = body.get("filename", "upload.png")

    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    arr = preprocess(img)

    logits = session.run(None, {INPUT_NAME: arr})[0][0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    prediction_id = str(uuid.uuid4())

    # Saliency heatmap → S3
    heat = occlusion_saliency(arr, pred_idx)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8))
    buf = io.BytesIO()
    heat_img.save(buf, format="PNG")
    heat_key = f"results/{prediction_id}.png"
    s3.put_object(Bucket=S3_BUCKET, Key=heat_key, Body=buf.getvalue(), ContentType="image/png")

    # Prediction history → DynamoDB
    record = {
        "prediction_id": prediction_id,
        "predicted_class": CLASSES[pred_idx],
        "confidence": str(round(float(probs[pred_idx]) * 100, 1)),
        "timestamp": str(int(time.time())),
        "filename": filename,
    }
    dynamo.put_item(Item=record)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "prediction_id": prediction_id,
            "predicted_class": CLASSES[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 1),
            "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)},
            "heatmap_url": f"s3://{S3_BUCKET}/{heat_key}",
        }),
    }
