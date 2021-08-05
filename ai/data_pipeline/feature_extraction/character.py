from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer

import numpy as np

"""
Considers features on a character level and measures the q-gram similarity
between the input strings
"""


def compute_q_gram_similarity(string1, string2, qval):
    """
    Measures the jaccard similarity between the character-based q-grams(n-grams).
    :param string1:  first sentence to compute q-grams for
    :param string2: second sentence for consideration
    :param qval: the value of 'q' in q-grams, e.g. 3, 4
    :return:  q_gram_similarity(float): jaccard similarity between the q-grams

    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/QgramTokenizer.html
    """
    q_gram_tokenizer = QgramTokenizer(qval=qval)
    q_grams1 = q_gram_tokenizer.tokenize(string1)
    q_grams2 = q_gram_tokenizer.tokenize(string2)
    return np.round(Jaccard().get_raw_score(q_grams1, q_grams2), 3)
