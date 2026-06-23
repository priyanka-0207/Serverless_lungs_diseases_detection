# Lambda container image based on the AWS Python base + ONNX Runtime.
# ONNX Runtime keeps the image small enough for the ECR free tier,
# where a full PyTorch image would be too large.

FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
