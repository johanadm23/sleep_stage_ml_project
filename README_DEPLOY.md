
Deployment Guide — Sleep Stage Classifier API

This document explains how to:

1. Build & test locally

2. Push Docker image to Docker Hub

3. Deploy on Google Cloud Run

4. Deploy on AWS ECS (Fargate)

1. Local build & test
Build Docker image
docker build -t sleep-stage-api .

Run
docker run --rm -p 8080:80 -v "%cd%/models:/app/models:ro" sleep-stage-api

Test
curl http://localhost:8080/

2. Push to Docker Hub
Login:
docker login

Tag:
docker tag sleep-stage-api YOUR_DOCKERHUB_USERNAME/sleep-stage-api:latest

Push:
docker push YOUR_DOCKERHUB_USERNAME/sleep-stage-api:latest

3. Deploy to Google Cloud Run
Enable Cloud Run
gcloud services enable run.googleapis.com

Deploy:
gcloud run deploy sleep-stage-api \
  --image="docker.io/YOUR_DOCKERHUB_USERNAME/sleep-stage-api:latest" \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated


Cloud Run gives you a public HTTPS URL:
https://sleep-stage-api-xxxxxx-uc.a.run.app

Call /predict exactly the same way.

4. Deploy on AWS ECS Fargate
Create ECS task with:

Container image: your Docker Hub image

Port mapping: 80 → 80

Memory: 512MB minimum

CPU: 256 minimum

Networking:

Public IP enabled

Security group: allow port 80 inbound

Run Task → open the public IP.

* Useful API test commands
curl -X POST https://YOUR_URL/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [0.12,0.34,0.45,0.09,0.001,0.5,-0.2,1.3]}"

* Updating the model

Retrain locally → replace artifacts inside models/ → rebuild Docker image:

docker build -t sleep-stage-api .
docker push YOUR_DOCKERHUB_USERNAME/sleep-stage-api:latest


Cloud Run / ECS will automatically pick up the new version when redeployed.

* Notes

FastAPI runs behind uvicorn inside the container

Models are stored in /app/models

If you want GPU deployment → use GCP or AWS GPU instances
