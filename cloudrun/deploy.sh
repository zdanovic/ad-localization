#!/usr/bin/env bash
set -euo pipefail

# Deploy Video Text Masker API to Cloud Run
# Prereqs: gcloud CLI authenticated and project selected, billing enabled

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=us-central1}"
: "${SERVICE_NAME:=video-text-masker}"
: "${AR_REPO:=video-api}"
: "${BUCKET_NAME:?Set BUCKET_NAME (target GCS bucket)}"
: "${TMP_DIR:=/tmp}"

SA_NAME="${SERVICE_NAME}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:latest"

echo "== Enable services"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iamcredentials.googleapis.com \
  videointelligence.googleapis.com \
  storage.googleapis.com \
  --project "${PROJECT_ID}"

echo "== Create Artifact Registry repo (if missing)"
gcloud artifacts repositories describe "${AR_REPO}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location "${REGION}" \
  --description "Containers for video text masker"

echo "== Create service account ${SA_EMAIL} (if missing)"
gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create "${SA_NAME}" --project "${PROJECT_ID}" \
  --description="Cloud Run SA for ${SERVICE_NAME}" --display-name="${SERVICE_NAME}"

echo "== Grant roles to service account"
# Project-level broad perms for dev (storage + video annotate)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/editor" >/dev/null
# Allow signing for GCS signed URLs
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/iam.serviceAccountTokenCreator" >/dev/null

echo "== Create GCS bucket (if missing): gs://${BUCKET_NAME}"
gsutil ls -b "gs://${BUCKET_NAME}" >/dev/null 2>&1 || \
gsutil mb -l "${REGION}" -b on "gs://${BUCKET_NAME}"

echo "== Grant bucket-level permissions"
gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" "gs://${BUCKET_NAME}"

echo "== Build & push image: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}"

echo "== Deploy to Cloud Run: ${SERVICE_NAME} (${REGION})"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --service-account "${SA_EMAIL}" \
  --max-instances=1 \
  --min-instances=0 \
  --no-cpu-throttling \
  --concurrency=8 \
  --memory=1Gi \
  --timeout=600 \
  --set-env-vars "GCS_BUCKET=${BUCKET_NAME},TMP_DIR=${TMP_DIR},ENV=prod,GCP_PROJECT=${PROJECT_ID}"

echo "== Done. Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)'

cat <<EOF

Next steps:
- Health check: curl "+ above URL + "/healthz
- Create job:
  curl -s -X POST \
    -F "file=@/path/to/sample.mp4" \
    -F 'languages=["en","ru"]' \
    "<SERVICE_URL>/videos"
EOF
