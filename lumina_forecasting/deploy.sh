#!/usr/bin/env bash
# Deploy the Lumina forecasting pipeline as a Cloud Run Job.
# Run from the PARENT directory of lumina_forecasting/ (the build context).
#
# Prereqs: gcloud CLI installed + `gcloud auth login` done. Edit the vars below.
set -euo pipefail

# ---- EDIT THESE ------------------------------------------------------------
PROJECT_ID="your-project-id"          # your GCP project id
REGION="us-central1"                  # region for Artifact Registry + Cloud Run
BUCKET="your-project-id-lumina"       # Cloud Storage bucket (must be globally unique)
REPO="lumina-repo"                    # Artifact Registry repo name
IMAGE="lumina-forecasting"            # image name
JOB="lumina-pipeline"                 # Cloud Run Job name
# ---------------------------------------------------------------------------

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

echo "==> Setting project"
gcloud config set project "${PROJECT_ID}"

echo "==> Enabling required APIs"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com

echo "==> Creating Artifact Registry repo (ignore error if it exists)"
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  --description="Lumina images" || true

echo "==> Creating bucket + uploading the dataset (ignore error if bucket exists)"
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" || true
gcloud storage cp Lumina_Markets_Dataset.xlsx "gs://${BUCKET}/data/Lumina_Markets_Dataset.xlsx"

echo "==> Building the image with Cloud Build (no local Docker needed)"
# Build context is THIS dir; Dockerfile lives inside the package folder.
gcloud builds submit --tag "${IMAGE_URI}" --gcs-log-dir="gs://${BUCKET}/build-logs" \
  --substitutions=_NONE=_ . \
  || gcloud builds submit -f lumina_forecasting/Dockerfile --tag "${IMAGE_URI}" .

echo "==> Creating (or updating) the Cloud Run Job"
gcloud run jobs deploy "${JOB}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --memory 2Gi --cpu 2 --task-timeout 3600 \
  --set-env-vars "LUMINA_EXCEL_PATH=gs://${BUCKET}/data/Lumina_Markets_Dataset.xlsx,LUMINA_ARTIFACT_DIR=gs://${BUCKET}/artifacts,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"

echo "==> Executing the job once now"
gcloud run jobs execute "${JOB}" --region "${REGION}" --wait

echo "==> Done. View logs:"
echo "    gcloud run jobs executions list --job ${JOB} --region ${REGION}"
echo "    gcloud logging read 'resource.type=cloud_run_job' --limit 50 --freshness=1h"
