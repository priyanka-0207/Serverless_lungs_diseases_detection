# Architecture & Build

The system is a serverless inference pipeline. The model is trained offline, exported to ONNX, and served from a containerized Lambda behind a REST API. Everything is provisioned in `us-east-1` and stays within AWS free tier limits.

## Build phases

### 1. DynamoDB table
- Table: `lung-predictions`
- Partition key: `prediction_id` (String)
- On-demand billing
- Stores: predicted class, confidence, timestamp, filename

### 2. S3 bucket
- Bucket holds `model.onnx` and a `results/` folder for generated heatmaps
- Region: `us-east-1`

### 3. ECR repository
- Private repo hosting the Lambda container image
- Image is built on an ONNX Runtime base to stay small

### 4. IAM role (least privilege)
- Policies: S3 read, DynamoDB write, Lambda basic execution
- Inline policy scoped to `s3:PutObject` on the `results/*` prefix only

### 5. Build and push the container
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-uri>
cd lambda_function
docker build --platform linux/amd64 -t lung-classifier .
docker tag lung-classifier:latest <ecr-uri>:latest
docker push <ecr-uri>:latest
```

### 6. Lambda function
- Package type: container image (from ECR)
- Memory: 1024 MB
- Timeout: 30 s
- Env vars: `S3_BUCKET`, `DYNAMO_TABLE`

### 7. API Gateway
- REST API exposing `POST /predict`, integrated with the Lambda function
- Stage: `prod`

### 8. CloudWatch
- Dashboard tracking Lambda invocations, average duration, and error count
- Logs at `/aws/lambda/lung-classifier`

## End-to-end request flow

1. User uploads a chest X-ray in the Streamlit app
2. App base64-encodes the image and POSTs to API Gateway `/predict` (~1 s)
3. Lambda loads the ONNX model from S3 and runs EfficientNet-B2 inference (~10 s)
4. Lambda generates a Grad-CAM heatmap and writes it to S3 `results/` (~2 s)
5. Prediction record written to DynamoDB (~0.2 s)
6. Streamlit displays class, confidence bar chart, and heatmap overlay (~1 s)

## Design tradeoffs

- **ONNX over PyTorch in Lambda:** the full PyTorch image was too large for the ECR free tier, so the model is exported to ONNX and served with ONNX Runtime, dropping the image well under the limit.
- **ONNX in Lambda over a always-on server:** serverless inference means no idle cost and automatic scaling, at the price of cold-start latency on the first request.
- **Grad-CAM for explainability:** the heatmap is generated from a convolutional layer's activations and gradients, then overlaid on the original X-ray so a reviewer can see which lung regions influenced the prediction.
