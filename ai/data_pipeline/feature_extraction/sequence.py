from py_stringmatching.similarity_measure.bag_distance import BagDistance
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.needleman_wunsch import NeedlemanWunsch
from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman

import numpy as np

"""
Sequence-based measure to consider order of words. All measures are some form of edit-based
distance between the two input strings
"""


def compute_bag_similarity(string1, string2):
    """
    Computes the normalized bag similarity between two strings.
    :param string1:
    :param string2:
    :return:

    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/BagDistance.html
    """
    return np.round(BagDistance().get_sim_score(string1, string2), 3)


def compute_lavenshtein_similarity(string1, string2):
    """
    Computes the normalized Levenshtein similarity score between two strings.
    :param string1:
    :param string2:
    :return:

    https://anhaidgroup.github.io/py_stringmatching/v0.3.x/Levenshtein.html
    """
    return np.round(Levenshtein().get_sim_score(string1, string2), 3)


def compute_needleman_wunsch_similarity(string1, string2):
    """
    Computes the raw Needleman-Wunsch score between two strings which is a extension of
    the Lavenshtein distance by performing dynamic global se- quence alignment
    :param string1:
    :param string2:
    :return:
    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/NeedlemanWunsch.html
    """
    return np.round(NeedlemanWunsch().get_raw_score(string1, string2), 3)


def compute_smith_waterman_similarity(string1, string2):
    """
    Computes the raw Smith-Waterman score between two strings.
    :param string1:
    :param string2:
    :return:

    Reference:
        https://anhaidgroup.github.io/py_stringmatching/v0.3.x/SmithWaterman.html
    """
    return np.round(SmithWaterman().get_raw_score(string1, string2), 3)


def compute_sequence_based_features(string1, string2):
    bag_sim = compute_bag_similarity(string1, string2)
    lavenshtein_sim = compute_lavenshtein_similarity(string1, string2)
    needleman_wunsch_sim = compute_needleman_wunsch_similarity(string1, string2)
    smith_waterman_sim = compute_smith_waterman_similarity(string1, string2)
    return [bag_sim, lavenshtein_sim, needleman_wunsch_sim, smith_waterman_sim]
