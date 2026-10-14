FROM public.ecr.aws/lambda/python:3.12
RUN pip install --no-cache-dir onnxruntime==1.30.0 boto3 tokenizers==0.23.2
COPY model_io.py app.py tokenizer.json ${LAMBDA_TASK_ROOT}/
CMD ["app.handler"]
