#!/bin/bash

REPO_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker stop reporter-ml-pipeline
docker rm reporter-ml-pipeline
docker build -t reporter-ml-pipeline -f ./ai/Dockerfile .


if [[ "$*" =~ "positive_train_samples" ]]; then
  dvc run -n sync_bio_sent \
  -d ai/data/positive_train_samples/__init__.py \
  -o data/df_postive_w_id_for_train_and_stats.csv \
  docker run -ti \
  --env-file "$(pwd)"/.env \
  --mount type=bind,src="$(pwd)"/data,dst=/app/data \
  --mount type=bind,src="$(pwd)"/ai,dst=/app/ai \
  --mount type=bind,src="$(pwd)"/shared,dst=/app/shared \
  --name reporter-ml-pipeline \
  reporter-ml-pipeline \
  -c "python -m ai.data.positive_train_samples ."
fi

if [[ "$*" =~ "sync_bio_sent" ]]; then
  dvc run -n sync_bio_sent \
  -d ai/data/sync_bio_sent/__init__.py \
  -o data/bioSent2Vec.bin \
  docker run -ti \
  --env-file "$(pwd)"/.env \
  --mount type=bind,src="$(pwd)"/data,dst=/app/data \
  --mount type=bind,src="$(pwd)"/ai,dst=/app/ai \
  --mount type=bind,src="$(pwd)"/shared,dst=/app/shared \
  --name reporter-ml-pipeline \
  reporter-ml-pipeline \
  -c "python -m ai.data.sync_bio_sent ."
fi