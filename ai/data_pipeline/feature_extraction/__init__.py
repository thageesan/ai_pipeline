from ai.data_pipeline.embed_snippets_biosent import EmbeddedSentencesBioSent
from ai.data_pipeline.embed_snippets_umlsbert import EmbeddedSentencesUMLSBert

from ai.data_pipeline.feature_extraction.token import compute_token_based_features
from ai.data_pipeline.feature_extraction.character import compute_q_gram_similarity
from ai.data_pipeline.feature_extraction.sequence import compute_sequence_based_features
from ai.data_pipeline.feature_extraction.semantic import SemanticBasedFeatures
from ai.data_pipeline.feature_extraction.entity import compute_entity_similarity


from shared.tools.os import getenv
from shared.tools.utils import pd
from shared.tools.utils.text import clean_sentence, tokenize_sentence

from numpy import load, array


def deserialize(corpus_file_path):
    corpus = load(corpus_file_path)
    return corpus


def extract_features(samples, corpus_list, columns, embedded_snippets):
    x = []
    y = []

    for _, row in samples.iterrows():
        features = []
        finding = clean_sentence(row[columns[0]])
        snippet = clean_sentence(row[columns[1]])

        tokenized_finding = tokenize_sentence(snippet)
        tokenized_snippet = tokenize_sentence(finding)

        if not tokenized_snippet or not tokenized_finding:
            continue

        features.extend(
            compute_token_based_features(corpus_list=corpus_list, list1=tokenized_snippet, list2=tokenized_finding))
        features.extend(compute_q_gram_similarity(string1=finding, string2=snippet, qval=3))
        features.extend(compute_q_gram_similarity(string1=finding, string2=snippet, qval=4))
        features.extend(compute_sequence_based_features(string1=finding, string2=snippet))
        features.extend(
            SemanticBasedFeatures(
                embedded_snippets=embedded_snippets,
                train_mode=True
            ).compute_semantic_based_features(
                string1=finding,
                string2=snippet
            )
        )
        features.extend(compute_entity_similarity(string1=finding, string2=snippet))
        x.append(features)
        y.append(row['label'])

    return array(x), array(y)


def app():
    columns = ['finding', 'snippet']
    data_path = getenv('DATA_FOLDER')
    biosent_snippet_file = getenv('EMBED_SNIPPETS_BIOSENT_FILE')
    uml_snippet_file = getenv('EMBED_SNIPPETS_UMLSBERT_FILE')
    corpus_file = getenv('CORPUS_FILE')
    training_samples_file = getenv('TRAINING_SAMPLES_FILE')

    embedded_biosent_snippets_file_path = f'{data_path}/{biosent_snippet_file}'
    embedded_uml_snippets_file_path = f'{data_path}/{uml_snippet_file}'

    training_samples = pd.read_parquet(f'{data_path}/{training_samples_file}')

    corpus = deserialize(corpus_file_path=f'{data_path}/{corpus_file}')

    embedded_biosent_snippets = EmbeddedSentencesBioSent()
    biosent_snippets = embedded_biosent_snippets.load_embedded_sentences(
        embedded_snippets_file_path=embedded_biosent_snippets_file_path)

    embedded_umlsbert_snippets = EmbeddedSentencesUMLSBert()
    umlsbert_snippets = embedded_umlsbert_snippets.load_embedded_sentences(
        embedded_snippets_file_path=embedded_uml_snippets_file_path)

    extract_features(samples=training_samples, corpus_list=corpus, columns=columns, )
