import re

from functools import lru_cache
from gensim.corpora.dictionary import Dictionary
import numpy as np
from pyemd import emd
from sent2vec import Sent2vecModel
from transformers import AutoModel, AutoTokenizer

from shared.tools.utils.text import convert_num_to_words


class SemanticBasedFeatures:
    """
    Computes the cosine similarity between embeddings of the input sentences
    and measures the similarity between the numbers in the two input strings
    using word mover's distance
    """

    def __init__(self):

        self.biosent2vec_model = Sent2vecModel()
        self.biosent2vec_model.load_model(model_path)

        self.snippets_biosent2vec_embed = snippets_biosent2vec_embed

        self.umlsbert_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')

        self.umlsbert_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

        special_tokens_dict = {'additional_special_tokens': ['xxx']}

        self.umlsbert_tokenizer.add_special_tokens(special_tokens_dict)
        self.umlsbert_model.resize_token_embeddings(len(self.umlsbert_tokenizer))

        self.snippets_umlsbert_embed = snippets_umlsbert_embed

    # used
    def compute_wmd_score(self, string1, string2):
        dictionary = Dictionary(documents=[string1, string2])
        vocab_len = len(dictionary)

        # Sets for faster look-up.
        docset1 = set(string1)
        docset2 = set(string2)

        # Compute distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if t1 not in docset1 or t2 not in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                t1_embed = self.compute_umls_bert_embed(t1)
                t2_embed = self.compute_umls_bert_embed(t2)
                normalized_embed_t1 = t1_embed / t1_embed.sum()
                normalized_embed_t2 = t2_embed / t2_embed.sum()
                distance_matrix[i, j] = np.sqrt(np.sum((normalized_embed_t1 - normalized_embed_t2) ** 2))

        if np.sum(distance_matrix) == 0:
            return 1

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            bag_of_words = dictionary.doc2bow(document)  # Word frequencies
            doc_len = len(document)
            for idx, freq in bag_of_words:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies
            return d

        # Compute nBOW representation of documents.
        d1 = nbow(string1)
        d2 = nbow(string2)

        # Compute WMD score
        return 1 - emd(d1, d2, distance_matrix)

    # used
    def computed_avg_wmd_score(self, list1, list2):
        wmd_scores = []
        longer_list = list1 if len(list1) > len(list2) else list2
        shorter_list = list1 if len(list1) < len(list2) else list2
        for d1 in longer_list:
            wmd_score = 0
            for d2 in shorter_list:
                score = self.compute_wmd_score([d1], [d2])
                if score >= wmd_score:
                    wmd_score = score
            wmd_scores.append(wmd_score)
        return np.mean(wmd_scores)

    # used
    @lru_cache(maxsize=100)
    def compute_biosent2vec_embed(self, string):
        string = convert_num_to_words(string=string)
        return self.biosent2vec_model.embed_sentence(string)

    # used
    def compute_biosent2vec_similarity(self, string1, string2):
        string1_embed = self.compute_biosent2vec_embed(string1)
        string2_embed = self.snippets_biosent2vec_embed[string2]
        biosent2vec_similarity = np.dot(np.squeeze(string1_embed), np.squeeze(string2_embed)) / (
                np.norm(string1_embed) * np.norm(string2_embed))
        return np.round(biosent2vec_similarity, 3)

    # used
    @lru_cache(maxsize=100)
    def compute_umls_bert_embed(self, string):
        string = convert_num_to_words(string=string)
        inputs = self.umlsbert_tokenizer(string, return_tensors='pt')
        return self.umlsbert_model(**inputs).last_hidden_state[0][0].detach().numpy()

    # used
    def compute_umlsbert_similarity(self, string1, string2):
        string1_embed = self.compute_umls_bert_embed(string1)

        string2_embed = self.snippets_umlsbert_embed[string2]
        umlsbert_similarity = np.dot(string1_embed, string2_embed) / (np.norm(string1_embed) * np.norm(string2_embed))
        return np.round(umlsbert_similarity, 3)

    # used
    def compute_word_movers_distance_score(self, string1, string2):
        string1_has_numbers = re.findall(r'\d+\.\d+|\d+', string1)
        string2_has_numbers = re.findall(r'\d+\.\d+|\d+', string2)
        if string1_has_numbers and string2_has_numbers:
            string1_digits = [convert_num_to_words(d) for d in string1_has_numbers]
            string2_digits = [convert_num_to_words(d) for d in string2_has_numbers]
            return self.computed_avg_wmd_score(string1_digits, string2_digits)  # should this just be on the numbers?

        elif not string1_has_numbers and not string2_has_numbers:
            return 1
        return 0

    # used
    def compute_semantic_based_features(self, string1, string2):
        biosent2vec_sim = self.compute_biosent2vec_similarity(string1, string2)
        umlsbert_sim = self.compute_umlsbert_similarity(string1, string2)
        word_movers_dist = self.compute_word_movers_distance_score(string1, string2)
        return [biosent2vec_sim, umlsbert_sim, word_movers_dist]
