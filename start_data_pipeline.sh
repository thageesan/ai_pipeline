#!/bin/bash

if [[ "$*" =~ "migrate_snippet_database" ]]; then
  docker-compose run ml.thageesan "python -m ai.data.snippet_database ."
fi

if [[ "$*" =~ "migrate_negative_train_samples" ]]; then
  docker-compose run ml.thageesan "python -m ai.data.negative_train_samples ."
fi


if [[ "$*" =~ "migrate_positive_train_samples" ]]; then
  docker-compose run ml.thageesan "python -m ai.data.positive_train_samples ."
fi

if [[ "$*" =~ "migrate_bio_sent" ]]; then
  docker-compose run ml.thageesan "python -m ai.data.sync_bio_sent ."
fi

if [[ "$*" =~ "download_umlsbert" ]]; then
  docker-compose run ml.thageesan "python -m ai.data.download_umlsbert ."
fi


if [[ "$*" =~ "sync_bio_sent" ]]; then
  dvc run -n sync_bio_sent \
  -d s3://ezra-ml-dvc/reporter/bioSent2Vec.bin \
  -o data/bioSent2Vec.bin \
  aws s3 cp s3://ezra-ml-dvc/reporter/bioSent2Vec.bin
fi

if [[ "$*" =~ "sync_negative_train_samples" ]]; then
  dvc run -n sync_negative_train_samples \
  -d s3://ezra-ml-dvc/reporter/df_negative_for_train_and_stats.csv \
  -o data/df_negative_for_train_and_stats.csv \
  aws s3 cp s3://ezra-ml-dvc/reporter/df_negative_for_train_and_stats.csv ./data/df_negative_for_train_and_stats.csv
fi

if [[ "$*" =~ "sync_positive_train_samples" ]]; then
  dvc run -n sync_positive_train_samples \
  -d s3://ezra-ml-dvc/reporter/df_positive_w_id_for_train_and_stats.csv \
  -o data/df_positive_w_id_for_train_and_stats.csv \
  aws s3 cp s3://ezra-ml-dvc/reporter/df_positive_w_id_for_train_and_stats.csv ./data/df_positive_w_id_for_train_and_stats.csv
fi

if [[ "$*" =~ "sync_snippet_database" ]]; then
  dvc run -n sync_snippet_database \
  -d s3://ezra-ml-dvc/reporter/cleaned_snippets_with_org_name_new_rows.csv \
  -o data/cleaned_snippets_with_org_name_new_rows.csv \
  aws s3 cp s3://ezra-ml-dvc/reporter/cleaned_snippets_with_org_name_new_rows.csv ./data/cleaned_snippets_with_org_name_new_rows.csv
fi