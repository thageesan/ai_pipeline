def compute_entity_similarity(string1, string2):
    """
    Checks if one string is a subset of the other, especially useful for short snippet titles
    where the title is just the name of a condition
    :param string1:
    :param string2:
    :return:
    """
    if string1 in string2 or string2 in string1:
        return 1
    return 0
