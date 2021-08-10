#!/bin/bash

if [[ "$*" =~ "migrate_snippet_database" ]]; then
  docker-compose run ml.thageesan "python -m ai.migration.snippet_database ."
fi

if [[ "$*" =~ "migrate_negative_train_samples" ]]; then
  docker-compose run ml.thageesan "python -m ai.migration.negative_train_samples ."
fi


if [[ "$*" =~ "migrate_positive_train_samples" ]]; then
  docker-compose run ml.thageesan "python -m ai.migration.positive_train_samples ."
fi

if [[ "$*" =~ "migrate_bio_sent" ]]; then
  docker-compose run ml.thageesan "python -m ai.migration.sync_bio_sent ."
fi

if [[ "$*" =~ "download_umlsbert" ]]; then
  docker-compose run ml.thageesan "python -m ai.migration.download_umlsbert ."
fi


if [[ "$*" =~ "migrate_umlsbert" ]]; then
  docker-compose run ml.thageesan "python -m ai.migration.migrate_umlsbert ."
fi


if [[ "$*" =~ "sync_bio_sent" ]]; then
  dvc run -n sync_bio_sent \
  -d ai/migration/sync_bio_sent/__init__.py \
  -d ai/migration/sync_bio_sent/__main__.py \
  -o data/bioSent2Vec.bin \
  -p params.py:BIOSENT_FILE_NAME \
  docker-compose run ml.thageesan "python -m ai.migration.sync_bio_sent ."
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
  -d ai/migration/download_umlsbert/__init__.py \
  -d ai/migration/download_umlsbert/__main__.py \
  -o data/UMLS/config.json \
  -o data/UMLS/pytorch_model.bin \
  -o data/UMLS/vocab.txt \
  -p params.py:UMLS_MODEL_NAME \
  docker-compose run ml.thageesan "python -m ai.migration.download_umlsbert ."
fi

if [[ "$*" =~ "embed_umlsbert_snippets" ]]; then
  dvc run -n embed_umlsbert_snippets \
  -d data/UMLSBert/config.json \
  -d data/UMLSBert/pytorch_model.bin \
  -d data/UMLSBert/vocab.txt \
  -d data/cleaned_snippets_with_org_name_new_rows.csv \
  -d ai/data_pipeline/embed_snippets_umlsbert/__init__.py \
  -d ai/data_pipeline/embed_snippets_umlsbert/__main__.py \
  -o data/embed_snippets_umls.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_snippets_umlsbert ."
fi

if [[ "$*" =~ "embed_biosent_snippets" ]]; then
  dvc run -n embed_biosent_snippets \
  -d data/cleaned_snippets_with_org_name_new_rows.csv \
  -d ai/data_pipeline/embed_snippets_biosent/__init__.py \
  -d ai/data_pipeline/embed_snippets_biosent/__main__.py \
  -o data/embed_snippets_biosent.parquet \
  -p params.py:BIOSENT_FILE_NAME \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_snippets_biosent ."
fi

if [[ "$*" =~ "generate_training_samples" ]]; then
  dvc run -n generate_training_samples \
  -d data/df_positive_w_id_for_train_and_stats.csv \
  -d data/df_negative_for_train_and_stats.csv \
  -d ai/data_pipeline/generate_training_samples/__init__.py \
  -d ai/data_pipeline/generate_training_samples/__main__.py \
  -o data/training_samples.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.generate_training_samples ."
fi

if [[ "$*" =~ "generate_corpus" ]]; then
  dvc run -n generate_corpus \
  -d data/cleaned_snippets_with_org_name_new_rows.csv \
  -d ai/data_pipeline/generate_corpus/__init__.py \
  -d ai/data_pipeline/generate_corpus/__main__.py \
  -o data/corpus.npy \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.generate_corpus ."
fi


if [[ "$*" =~ "extract_features_for_training" ]]; then
  dvc run -n extract_features_for_training \
  -d data/corpus.npy \
  -d data/embed_finding_umls.parquet \
  -d data/embed_finding_biosent.parquet \
  -d data/embed_snippets_umls.parquet \
  -d data/embed_snippets_biosent.parquet \
  -d data/embed_numbers_umls.parquet \
  -d data/training_samples.parquet \
  -d ai/data_pipeline/feature_extraction/__init__.py \
  -d ai/data_pipeline/feature_extraction/__main__.py \
  -o data/feature_extraction.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.feature_extraction ."
fi

if [[ "$*" =~ "embed_umlsbert_finding" ]]; then
  dvc run -n embed_umlsbert_finding \
  -d data/training_samples.parquet \
  -d ai/data_pipeline/embed_finding_umlsbert/__init__.py \
  -d ai/data_pipeline/embed_finding_umlsbert/__main__.py \
  -o data/embed_finding_umls.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_finding_umlsbert ."
fi

if [[ "$*" =~ "embed_biosent_finding" ]]; then
  dvc run -n embed_biosent_finding \
  -d data/training_samples.parquet \
  -d ai/data_pipeline/embed_finding_biosent/__init__.py \
  -d ai/data_pipeline/embed_finding_biosent/__main__.py \
  -o data/embed_finding_biosent.parquet \
  -p params.py:BIOSENT_FILE_NAME \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_finding_biosent ."
fi

if [[ "$*" =~ "embed_number_in_finding" ]]; then
  dvc run -n embed_number_in_finding \
  -d data/training_samples.parquet \
  -d ai/data_pipeline/embed_numbers_umls/__init__.py \
  -d ai/data_pipeline/embed_numbers_umls/__main__.py \
  -o data/embed_numbers_umls.parquet \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.embed_numbers_umls ."
fi


if [[ "$*" =~ "train_test_split" ]]; then
  dvc run -n train_test_split \
  -d data/feature_extraction_training.parquet \
  -d ai/data_pipeline/train_test_split/__init__.py \
  -d ai/data_pipeline/train_test_split/__main__.py \
  -o data/training_set.parquet \
  -o data/testing_set.parquet \
  -p params.py:TestTrainSetConfig \
  docker-compose run ml.thageesan "python -m ai.data_pipeline.train_test_split ."
fi

if [[ "$*" =~ "train_model" ]]; then
  dvc run -n train_model \
  -d data/training_set.parquet \
  -d ai/training/__init__.py \
  -d ai/training/__main__.py \
  -o data/xgb.model \
  -p params.py:XGBTrainConfig \
  docker-compose run ml.thageesan "python -m ai.training ."
fi

if [[ "$*" =~ "test_model" ]]; then
  dvc run -n test_model \
  -d data/testing_set.parquet \
  -d ai/test/__init__.py \
  -d ai/test/__main__.py \
  -M data/metrics/test_results.json \
  docker-compose run ml.thageesan "python -m ai.test ."
fi

if [[ "$*" =~ "playground" ]]; then
  docker-compose run ml.thageesan "python -m ai.test ."
fi