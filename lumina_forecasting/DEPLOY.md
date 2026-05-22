# Deploying the Lumina pipeline to GCP (Cloud Run Job)

This hosts **all nodes** on GCP. You do not deploy nodes individually — the
orchestrator already runs every node in sequence, so the whole package is
shipped as one container image and run as a single Cloud Run **Job** (a batch
job that runs to completion and exits, not a web service).

```
  Cloud Storage (bucket)              Cloud Run Job (your container)
  ├── data/Lumina_...xlsx   ──read──▶ data_agent → ... → summarizer
  └── artifacts/            ◀─write── predictions, run_summary
```

## What changes in the code
Nothing in the nodes. Two small things, already done:
- `data_io.py` now reads/writes `gs://` paths transparently (local paths still
  work unchanged).
- `config.py` reads `LUMINA_EXCEL_PATH` and `LUMINA_ARTIFACT_DIR` from env vars,
  so the same image works for any bucket without rebuilding.

## GCP services (APIs) to enable
The deploy script enables these for you, but for reference:
- **Cloud Run API** (`run.googleapis.com`) — runs the job
- **Cloud Build API** (`cloudbuild.googleapis.com`) — builds the image in the
  cloud, so you don't need Docker installed locally
- **Artifact Registry API** (`artifactregistry.googleapis.com`) — stores the image
- **Vertex AI API** (`aiplatform.googleapis.com`) — Gemini narrative (already on)
- **Cloud Storage** (`storage.googleapis.com`) — the data + artifacts

## One-time prerequisites
1. Install the gcloud CLI and run `gcloud auth login`.
2. Make sure billing/trial credit is linked to your project.

## Deploy (the easy way)
From the **parent** directory of `lumina_forecasting/`:
```bash
# edit the variables at the top of the script first
bash lumina_forecasting/deploy.sh
```
That script: sets the project, enables APIs, creates an Artifact Registry repo,
creates a bucket and uploads the dataset, builds the image with Cloud Build,
creates the Cloud Run Job with the right env vars, and runs it once.

## Deploy (manual, if you prefer step-by-step)
```bash
# 0. variables
export PROJECT_ID=your-project-id
export REGION=us-central1
export BUCKET=${PROJECT_ID}-lumina
export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/lumina-repo/lumina-forecasting:latest

# 1. project + APIs
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com storage.googleapis.com

# 2. image registry
gcloud artifacts repositories create lumina-repo \
  --repository-format=docker --location=$REGION

# 3. bucket + data
gcloud storage buckets create gs://$BUCKET --location=$REGION
gcloud storage cp Lumina_Markets_Dataset.xlsx gs://$BUCKET/data/Lumina_Markets_Dataset.xlsx

# 4. build image (Cloud Build reads lumina_forecasting/Dockerfile)
gcloud builds submit -f lumina_forecasting/Dockerfile --tag $IMAGE_URI .

# 5. create the job
gcloud run jobs deploy lumina-pipeline \
  --image $IMAGE_URI --region $REGION \
  --memory 2Gi --cpu 2 --task-timeout 3600 \
  --set-env-vars LUMINA_EXCEL_PATH=gs://$BUCKET/data/Lumina_Markets_Dataset.xlsx,LUMINA_ARTIFACT_DIR=gs://$BUCKET/artifacts,GOOGLE_CLOUD_PROJECT=$PROJECT_ID

# 6. run it
gcloud run jobs execute lumina-pipeline --region $REGION --wait
```

## Add the Gemini narrative on GCP
The job's service account needs the **Vertex AI User** role, then set the
backend to vertex. Either set `LLM_BACKEND="vertex"` in config.py before
building, or add `--args="--gemini"` plus an env var override. Grant the role:
```bash
PROJNUM=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJNUM}-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

## Schedule it nightly (optional)
```bash
gcloud scheduler jobs create http lumina-nightly \
  --location=$REGION --schedule="0 2 * * *" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/lumina-pipeline:run" \
  --http-method=POST \
  --oauth-service-account-email="${PROJNUM}-compute@developer.gserviceaccount.com"
```

## Check results
```bash
# execution status
gcloud run jobs executions list --job lumina-pipeline --region $REGION
# logs (the summary brief prints here)
gcloud logging read 'resource.type=cloud_run_job' --limit 80 --freshness=1h
# artifacts written back to the bucket
gcloud storage ls gs://$BUCKET/artifacts/
```

## Cost note
On the free trial this is a few cents per run: a Cloud Run Job that runs ~2
minutes with 2 vCPU / 2 GiB, plus negligible storage. The credit is far more
than you'll use.

## Common errors
- **`PermissionDenied` / 403 on build** → an API isn't enabled (step 1) or
  billing isn't linked.
- **`Could not determine credentials`** for Gemini → grant the Vertex AI User
  role (above); on GCP the job uses its service account, not your laptop login.
- **Job OOM / killed** → bump `--memory 4Gi`. At full 15k-SKU scale you'd move
  to a Vertex AI Custom Job with a bigger machine; the image is identical.
- **Image too big / slow build** → the `.dockerignore` already excludes data,
  artifacts, and caches; keep large files out of the build context.
