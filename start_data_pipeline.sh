#!/bin/bash

if [[ "$*" =~ "migrate_snippet_database" ]]; then
  docker-compose run ml.thageesan "python -m ai.data_pipeline.snippet_database ."
fi

if [[ "$*" =~ "migrate_negative_train_samples" ]]; then
  docker-compose run ml.thageesan "python -m ai.data_pipeline.negative_train_samples ."
fi


if [[ "$*" =~ "migrate_positive_train_samples" ]]; then
  docker-compose run ml.thageesan "python -m ai.data_pipeline.positive_train_samples ."
fi

if [[ "$*" =~ "migrate_bio_sent" ]]; then
  docker-compose run ml.thageesan "python -m ai.data_pipeline.sync_bio_sent ."
fi

if [[ "$*" =~ "download_umlsbert" ]]; then
  docker-compose run ml.thageesan "python -m ai.data_pipeline.download_umlsbert ."
fi


if [[ "$*" =~ "migrate_umlsbert" ]]; then
  docker-compose run ml.thageesan "python -m ai.data_pipeline.migrate_umlsbert ."
fi


if [[ "$*" =~ "sync_bio_sent" ]]; then
  dvc run -n sync_bio_sent \
  -d s3://ezra-ml-dvc/reporter/bioSent2Vec.bin \
  -o data/bioSent2Vec.bin \
  aws s3 cp s3://ezra-ml-dvc/reporter/bioSent2Vec.bin ./data/bioSent2Vec.bin
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

if [[ "$*" =~ "sync_uml_sbert" ]]; then
  dvc run -n sync_uml_sbert \
  -d s3://ezra-ml-dvc/reporter/UMLSBert/config.json \
  -d s3://ezra-ml-dvc/reporter/UMLSBert/pytorch_model.bin \
  -d s3://ezra-ml-dvc/reporter/UMLSBert/vocab.txt \
  -o data/UMLSBert/config.json \
  -o data/UMLSBert/pytorch_model.bin \
  -o data/UMLSBert/vocab.txt \
  aws s3 cp s3://ezra-ml-dvc/reporter/UMLSBert  ./data/UMLSBert --recursive
fi

if [[ "$*" =~ "embed_umlsbert_snippets" ]]; then
  dvc run -n embed_umlsbert_snippets \
  -d data/UMLSBert/config.json \
  -d data/UMLSBert/pytorch_model.bin \
  -d data/UMLSBert/vocab.txt \
  -d data/cleaned_snippets_with_org_name_new_rows.csv \
  -d ai/data_pipeline/embed_snippets_umlsbert/__init__.py \
  -d ai/data_pipeline/embed_snippets_umlsbert/__main__.py \
  -o data/embed_snippets_umlsbert.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_snippets_umlsbert ."
fi

if [[ "$*" =~ "embed_biosent_snippets" ]]; then
  dvc run -n embed_biosent_snippets \
  -d data/bioSent2Vec.bin \
  -d data/cleaned_snippets_with_org_name_new_rows.csv \
  -d ai/data_pipeline/embed_snippets_biosent/__init__.py \
  -d ai/data_pipeline/embed_snippets_biosent/__main__.py \
  -o data/embed_snippets_biosent.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_snippets_biosent ."
fi