#!/bin/bash

if [[ "$*" =~ "snippet_database" ]]; then
  dvc run -n snippet_database \
  -d ai/data/snippet_database/__init__.py \
  -o data/cleaned_snippets_with_org_name_new_rows.csv \
  docker-compose run ml.thageesan python -m ai.data.snippet_database .
fi

if [[ "$*" =~ "negative_train_samples" ]]; then
  dvc run -n negative_train_samples_train_samples \
  -d ai/data/negative_train_samples/__init__.py \
  -o data/df_negative_for_train_and_stats.csv \
  docker-compose run ml.thageesan python -m ai.data.negative_train_samples .
fi


if [[ "$*" =~ "positive_train_samples" ]]; then
  dvc run -n positive_train_samples \
  -d ai/data/positive_train_samples/__init__.py \
  -o data/df_positive_w_id_for_train_and_stats.csv \
  docker-compose run ml.thageesan python -m ai.data.positive_train_samples .
fi

if [[ "$*" =~ "sync_bio_sent" ]]; then
  dvc run -n sync_bio_sent \
  -d ai/data/sync_bio_sent/__init__.py \
  -o data/bioSent2Vec.bin \
  docker-compose run ml.thageesan python -m ai.data.sync_bio_sent .
fi