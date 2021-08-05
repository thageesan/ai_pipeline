from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.generalized_jaccard import GeneralizedJaccard
from py_stringmatching.similarity_measure.dice import Dice

import numpy as np

"""
Functions used to extract features at the word level.
"""


def compute_tf_idf_similarity(corpus_list, list1, list2):
    """
    TF/IDF measure on two input strings (using get_raw_score or get_sim_score), the corpus is taken to be the list of
    those two strings. TF/IDF score commonly used in information retrieval (IR) to find documents that are relevantto
    keyword queries. The intuition underlying the TF/IDF measure is that two strings are similar if they share
    distinguishing terms.

    https://anhaidgroup.github.io/py_stringmatching/v0.3.x/TfIdf.html

    :param corpus_list: is a list of strings, where each
        string has been tokenized into a list of tokens (that is, a bag of tokens)
    :param list1: An array of strings
    :param list2: An Array of strings
    :return:
    """
    return np.round(TfIdf(corpus_list).get_raw_score(list1, list2), 3)


def compute_jaccard_similarity(list1, list2):
    """
    Computes the Jaccard measure of the two lists of strings

    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/Jaccard.html

    :param list1:
    :param list2:
    :return:
    """
    return np.round(Jaccard().get_raw_score(list1, list2), 3)


def compute_generalized_jaccard_similarity(list1, list2, threshold=.5):
    """
    This similarity measure is softened version of the Jaccard measure.  It the real world many words are misspelled and
    in such cases the Generalized Jaccard will promote matching in such cases.

    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/GeneralizedJaccard.html

    :param list1:
    :param list2:
    :param threshold:
    :return:
    """
    return np.round(GeneralizedJaccard(threshold=threshold).get_raw_score(list1, list2), 3)


def compute_dice_similarity(list1, list2):
    """
    Returns the Dice score between two strings.

    The Dice similarity score is defined as twice the shared information (intersection) divided by sum of cardinalities.

    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/Dice.html
    """
    return np.round(Dice().get_raw_score(list1, list2), 3)


def compute_orchai_similarity(list1, list2):
    """
    Returns the Orchai score between two strings.

    The Orchai similarity score is defined as the shared information (intersection) divided by the square root
    of the multiplication of the cardinalities.

    """
    if not list1 or not list2:
        raise TypeError('One or both inputs are None')

    if not((isinstance(list1, list) and isinstance(list1, list)) or (isinstance(list2, set) and isinstance(list2, set))):
        raise TypeError('Input needs to be list or set')

    # if exact match return 1
    if list1 == list2:
        return 1

    # if one of the strings is empty return 0
    if list(list1) == [''] or list(list2) == ['']:
        return 0

    if not isinstance(list1, set):
        list1 = set(list1)
    if not isinstance(list2, set):
        list2 = set(list2)

    return np.round(float(len(list1 & list2)) / np.sqrt(float(len(list1) * len(list2))), 3)


def compute_token_based_features(corpus_list, list1, list2):
    """
    Calls all the defined methods in the class
    """
    jaccard_sim = compute_jaccard_similarity(list1, list2)
    generalized_jaccard_sim = compute_generalized_jaccard_similarity(list1, list2)
    dice_sim = compute_dice_similarity(list1, list2)
    orchai_sim = compute_orchai_similarity(list1, list2)
    tf_idf_sim = compute_tf_idf_similarity(corpus_list, list1, list2)
    return [jaccard_sim, generalized_jaccard_sim, dice_sim, orchai_sim, tf_idf_sim]
