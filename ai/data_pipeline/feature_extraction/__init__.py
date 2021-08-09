from ai.data_pipeline.embed_snippets_biosent import EmbeddedSnippetsBioSent
from ai.data_pipeline.embed_snippets_umlsbert import EmbeddedSnippetsUMLSBert
from ai.data_pipeline.embed_finding_biosent import EmbeddedFindingBioSent
from ai.data_pipeline.embed_finding_umlsbert import EmbeddedFindingUMLSBert

from ai.data_pipeline.feature_extraction.token import compute_token_based_features
from ai.data_pipeline.feature_extraction.character import compute_q_gram_similarity
from ai.data_pipeline.feature_extraction.sequence import compute_sequence_based_features
from ai.data_pipeline.feature_extraction.semantic import SemanticBasedFeatures
from ai.data_pipeline.feature_extraction.entity import compute_entity_similarity

from shared.tools.os import getenv
from shared.tools.utils import pd
from shared.tools.utils.text import clean_sentence, tokenize_sentence
from shared.embed_sentences.umls_bert_embeddor import UMLSEmbedder

from numpy import load, array


def deserialize(corpus_file_path):
    corpus = load(corpus_file_path, allow_pickle=True)
    return corpus


def extract_features(
        samples,
        corpus_list,
        columns,
        embedded_biosent_snippets,
        embedded_biosent_findings,
        embedded_umlsbert_snippets,
        embedded_umlsbert_findings,
        embedded_numbers_umlsbert
):
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
        features.extend([compute_q_gram_similarity(string1=finding, string2=snippet, qval=3)])
        features.extend([compute_q_gram_similarity(string1=finding, string2=snippet, qval=4)])
        features.extend(compute_sequence_based_features(string1=finding, string2=snippet))
        features.extend(
            SemanticBasedFeatures(
                embedded_biosent_finding=embedded_biosent_findings,
                embedded_biosent_snippets=embedded_biosent_snippets,
                embedded_umlsbert_finding=embedded_umlsbert_findings,
                embedded_umlsbert_snippets=embedded_umlsbert_snippets,
                embedded_numbers_umlsbert=embedded_numbers_umlsbert
            ).compute_semantic_based_features(
                string1=finding,
                string2=snippet
            )
        )
        features.extend([compute_entity_similarity(string1=finding, string2=snippet)])
        x.append(features)
        y.append(row['label'])

    data_fame = pd.DataFrame(y, columns=['y'])
    data_fame.insert(1, 'x', x, True)

    return data_fame


def app():
    columns = ['finding', 'snippet']
    data_path = getenv('DATA_FOLDER')
    biosent_snippet_file = getenv('EMBED_SNIPPETS_BIOSENT_FILE')
    biosent_finding_file = getenv('EMBED_FINDING_BIOSENT_FILE')
    uml_snippet_file = getenv('EMBED_SNIPPETS_UMLSBERT_FILE')
    uml_finding_file = getenv('EMBED_FINDING_UMLS_FILE')
    umls_numbers_file = getenv('EMBED_NUMBERS_FILE')
    corpus_file = getenv('CORPUS_FILE')
    training_samples_file = getenv('TRAINING_SAMPLES_FILE')
    feature_extraction_file = getenv('FEATURE_EXTRACTION_FILE')

    embedded_biosent_snippets_file_path = f'{data_path}/{biosent_snippet_file}'
    embedded_biosent_finding_file_path = f'{data_path}/{biosent_finding_file}'
    embedded_uml_snippets_file_path = f'{data_path}/{uml_snippet_file}'
    embedded_uml_finding_file_path = f'{data_path}/{uml_finding_file}'
    embedded_numbers_umls_file_path = f'{data_path}/{umls_numbers_file}'

    training_samples = pd.read_parquet(f'{data_path}/{training_samples_file}')

    corpus = deserialize(corpus_file_path=f'{data_path}/{corpus_file}').tolist()

    embedded_biosent_snippets = EmbeddedSnippetsBioSent()
    biosent_snippets = embedded_biosent_snippets.load_embedded_sentences(
        embedded_snippets_file_path=embedded_biosent_snippets_file_path
    )

    embedded_biosent_finding = EmbeddedFindingBioSent()

    biosent_findings = embedded_biosent_finding.load_embedded_sentences(
        embedded_finding_file_path=embedded_biosent_finding_file_path)

    embedded_umlsbert_snippets = EmbeddedSnippetsUMLSBert()
    umlsbert_snippets = embedded_umlsbert_snippets.load_embedded_sentences(
        embedded_snippets_file_path=embedded_uml_snippets_file_path
    )

    embedded_umlsbert_findings = EmbeddedFindingUMLSBert()
    umlsbert_findings = embedded_umlsbert_findings.load_embedded_sentences(
        embedded_finding_file_path=embedded_uml_finding_file_path)

    umlsbert_embedder = UMLSEmbedder()

    embedded_numbers_umlsbert = umlsbert_embedder.load_embedded_sentences(embedded_numbers_umls_file_path)

    data_frame = extract_features(
        samples=training_samples,
        corpus_list=corpus,
        columns=columns,
        embedded_biosent_snippets=biosent_snippets,
        embedded_biosent_findings=biosent_findings,
        embedded_umlsbert_snippets=umlsbert_snippets,
        embedded_umlsbert_findings=umlsbert_findings,
        embedded_numbers_umlsbert=embedded_numbers_umlsbert
    )

    data_frame.to_parquet(f'{data_path}/{feature_extraction_file}')
