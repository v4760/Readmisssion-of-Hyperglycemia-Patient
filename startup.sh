#!/bin/bash
set -e  # Exit immediately on any error

echo "[INFO] Updating system and installing prerequisites..."
apt-get update -y
apt-get install -y apt-transport-https ca-certificates curl software-properties-common git unzip

echo "[INFO] Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-get update -y
apt-get install -y docker-ce

echo "[INFO] Installing Docker Compose V2..."
DOCKER_COMPOSE_VERSION="v2.27.1"
mkdir -p ~/.docker/cli-plugins/
curl -SL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
docker compose version  # Verify installation

echo "[INFO] Verifying network connectivity..."
ping -c 3 github.com || echo "⚠️ GitHub not reachable, continuing anyway."

echo "[INFO] Cloning project repository..."
cd /home

# Add retry logic in case of failure
for attempt in {1..3}; do
  git clone https://github.com/mlops2025/Readmission-Prediction.git && break
  echo "⚠️ Clone attempt $attempt failed, retrying in 5s..."
  sleep 5
done

cd Readmission-Prediction || { echo "❌ Failed to cd into repo directory"; exit 1; }

echo "[INFO] Creating .env file from GCP instance metadata..."
cat <<EOF > .env
SMTP_USER=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/SMTP_USER" -H "Metadata-Flavor: Google")
SMTP_PASSWORD=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/SMTP_PASSWORD" -H "Metadata-Flavor: Google")
DB_HOST=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/DB_HOST" -H "Metadata-Flavor: Google")
DB_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/DB_NAME" -H "Metadata-Flavor: Google")
DB_PASS=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/DB_PASS" -H "Metadata-Flavor: Google")
DB_PORT=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/DB_PORT" -H "Metadata-Flavor: Google")
DB_USER=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/DB_USER" -H "Metadata-Flavor: Google")
GCP_BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCP_BUCKET_NAME" -H "Metadata-Flavor: Google")
GCP_PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCP_PROJECT_ID" -H "Metadata-Flavor: Google")
AIRFLOW_UID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/AIRFLOW_UID" -H "Metadata-Flavor: Google")
EOF




echo "[INFO] Service account key saved and environment variable set."

echo "[INFO] Creating required Airflow directories..."
mkdir -p dags logs plugins config data data/processed
chown -R "${USER}:${USER}" dags logs plugins config data

echo "[INFO] Writing GCP_SA_KEY to /path/to/your/config/key.json..."

curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCP_SA_KEY" \
  -o /home/Readmission-Prediction/config/key.json

echo "[INFO] GCP SA key written successfully ✅"
chmod 600 /home/Readmission-Prediction/config/key.json

echo "[INFO] Starting Airflow containers..."
export _PIP_ADDITIONAL_REQUIREMENTS="$(cat requirements.txt | tr '\n' ' ')"
docker compose up airflow-init
docker compose up -d
sudo chown -R 50000:0 .

echo "[✅ SUCCESS] Airflow stack is up and running!"
