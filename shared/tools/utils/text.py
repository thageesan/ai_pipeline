from nltk.corpus import stopwords
from nltk import download

import re
import string

from nltk import word_tokenize
from num2words import num2words


def clean_sentence(sentence):
    """
    Cleans a sentence from new lines extra spaces & punctuation

    Args:
        sentence(str): a sentence that has unnecessary characters

    Returns:
        cleaned_sentece(str): cleaned version of the input sentence
    """
    sentence = sentence.lower()
    sentence = re.sub(r'{.*}', '', sentence)
    sentence = re.sub(r'\(image(s)? \d+.*\)|\(series\s{0,}\d+:\s+image\s{0,}\d+\)', '',
                      sentence)  # remove (image(s) X, X)
    sentence = re.sub(r'(?<!\d)/', ' ', sentence)  # remove [], (), /(except for dates XX/XX/XXXX)
    sentence = re.sub(r'[\(\)\[\]]', ' ', sentence)
    sentence = re.sub(r'\d+\/\d+\/\d+|(?<!from)\d{4}', 'before', sentence)  # detect year
    sentence = re.sub(r'xx/xx/xxxx', 'before', sentence)
    sentence = re.sub(r'\.{3,}', '', sentence)
    sentence = re.sub(r'(\\n)+', ' ', sentence)
    sentence = re.sub(r'(\\t)+', ' ', sentence)
    sentence = ' '.join(sentence.split())
    sentence = re.sub(r'(?<!\d)[.,;:"](?!\d)', '', sentence)
    sentence = re.sub(r'_{2,}', '', sentence)
    sentence = re.sub(r'\bc(?=\d-?\d?\b)', 'cervical spine ', sentence)
    sentence = re.sub(r'\b[ls](?=\d-?\d?\b)', 'lumbar spine ', sentence)
    sentence = re.sub(r'\b[t](?=([3-9]|[1][1-2])-?\d?\b)', 'thoracic spine ', sentence)
    sentence = re.sub(r'\.$', '', sentence)
    return sentence.lstrip()


def tokenize_sentence(sentence):
    """
    Tokenize a cleaned sentence

    Args:
        sentence(str): a single sentence from a section/passage

    Returns:
        sentences_token(list): list of cleaned & lemmatised tokens of the input sentence
    """
    if sentence in UNINFORMATIVE_SENTENCES:  # used in extracting sentence pairs
        return []
    return [w for w in word_tokenize(sentence) if w not in stopwords_and_punc]


def convert_num_to_words(string):
    """
    Looks for digits in the string and changes them to words, e.g. 24 -> twenty-four

    Args:
        string(str): input string

    Returns:
        converted_string(str): same as input with the digits coverted to words
    """
    string = string.replace(',', '')
    string = re.sub(r'([0-9]+(\.[0-9]+)?)', r'\1 ',
                    string)  # in case the unit is sticking to the number, e.g. 2cm instead of 2 cm
    string_digits = re.findall(r'([0-9]+(\.[0-9]+)?)', string)
    for digit in string_digits:
        string = string.replace(digit[0], num2words(digit[0]))
    return ' '.join(string.split())


UNINFORMATIVE_SENTENCES = [
    'this ct exam has been performed using low dose protocols to limit radiation exposure to as low as reasonably achievable', 'this center is recognized and certified by the american college of radiology a designation awarded to centers who have demonstrated faculty competency\
                            clinical image excellence and radiation safety compliance requirements',
    'your mri shows the following findings and pi-rads score',
    'pirads score (probability of prostate cancer on a scale of 1 to 5:)', 'calcium score', 'cardiovascular',
    'coronary artery', 'aorta and branch arteries', 'non-cardiovascular', 'score', 'calcium', 'prostate', 'uterus',
    'spine', 'urinary bladder', '1', '2', '3', '4', '5', 'lumbar spine 1', 't2 hyperintensity']

download('stopwords')
download('punkt')
stopwords = set(stopwords.words('english'))
stopwords -= {'no', 'nor', 'not', 'none'}
stopwords_and_punc = list(stopwords)
stopwords_and_punc.extend(string.punctuation)