from shared.tools.data import pd, pandas_series_to_cudf_series
from shared.tools.execution_time import calculate_execution_time
from shared.tools.os import getenv, path

import pickle
import re
import string

from nltk.corpus import stopwords
from nltk import word_tokenize, download

data_folder = '/app/data/'
columns = ['finding', 'snippet']

ROOT_DIR = path.dirname(path.abspath(__file__))

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


def find_anomaly_ids(data_frames, columns):
    """
    This will examine the data frames (based on the columns passed) to see if the shuffling created an anomaly.  If
    detected it will return an array of ids
    :param data_frames:
    :param columns:
    :return:
    """
    data_frame_group_by = data_frames.groupby(list(data_frames[columns]))

    return [x[0] for x in data_frame_group_by.groups.values() if len(x) != 1]


def add_negative_samples(df_positive, rows_needed, columns):
    """
    Takes the dataframe with only positive examples and addes negative examples to it

    Args:
        df(dataframe): dataframe with columns: sentence_1, sentence_2 and label

    Returns:
        df(dataframe): df around double the size of input with negative examples
    """
    if rows_needed == 0:
        return df_positive

    df_sample = df_positive.sample(n=rows_needed).reset_index()

    df_mismatch = df_sample.sample(frac=1)
    df_mismatch['snippet'] = df_mismatch['snippet'].sample(frac=1).values
    df_mismatch['label'] = [0] * len(df_mismatch)

    df_temp = pd.concat([df_mismatch, df_sample]).reset_index(drop=True)

    idx = find_anomaly_ids(df_temp, columns)
    df_mismatch = df_mismatch.drop(idx)
    return pd.concat([df_positive, df_mismatch])


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


def prepare_data_frame(csv_file):
    data_frame = pd.read_csv(f'{data_folder}{csv_file}')

    for column in columns:
        if hasattr(data_frame[column], 'to_pandas'):
            panda_series = data_frame[column].to_pandas()
            panda_series = panda_series.map(lambda s: clean_sentence(s))
            data_frame[column] = pandas_series_to_cudf_series(pandas_series=panda_series)
        else:
            data_frame[column] = data_frame[column].map(lambda s: clean_sentence(s))

    return data_frame.drop_duplicates(columns).reset_index()


def extract_corpus(df, corpus_list, columns):
    """
    Creates a list of list of tokens for each sentence in sentence_pairs.
    Used for the tf-idf similarity measure

    Args:
        df(dataframe): dataframe with columns sentence_1, sentence_2 and label

    Returns:
        corpus_list(list of lists): words in each sentence to be considered as the corpus
    """

    if hasattr(df, 'to_pandas'):
        rows = df.to_pandas().iterrows()
    else:
        rows = df.iterrows()

    for _, row in rows:
        docs = [[row[c]] for c in columns]
        for doc in docs:
            cleaned_sentence = clean_sentence(doc[0])
            corpus_list.append(tokenize_sentence(cleaned_sentence))
    return corpus_list


def create_and_dump_corpus(df_postive, df_negative, snippets):
    corpus_list = []
    corpus_list.extend(extract_corpus(df_postive, corpus_list, columns))
    corpus_list.extend(extract_corpus(df_negative, corpus_list, columns))
    corpus_list.extend(extract_corpus(snippets, corpus_list, ['Name']))
    pickle.dump(corpus_list, open('{}/corpus.pkl'.format(data_folder), 'wb'))
    return corpus_list


def prepare_data(positive_csv_file, negative_csv_file, snippet_csv_file):
    data_frame_positive = prepare_data_frame(csv_file=positive_csv_file)
    data_frame_negative = prepare_data_frame(csv_file=negative_csv_file)
    data_frame_snippet = pd.read_csv(f'{data_folder}{snippet_csv_file}')

    data_frame = add_negative_samples(data_frame_positive,
                                      rows_needed=len(data_frame_positive) - len(data_frame_negative),
                                      columns=columns)
    data_frame = pd.concat(
        [
            data_frame[['finding', 'snippet', 'label']],
            data_frame_negative[['finding', 'snippet', 'label']]
        ]
    ).reset_index()

    if data_frame.isnull().sum().sum() != 0:
        raise ValueError('Stopped because there is at least one row with a null label')

    corpus = create_and_dump_corpus(data_frame_positive, data_frame_negative, data_frame_snippet)

    return corpus, data_frame


def app():
    positive_csv_file = getenv('POSITIVE_CSV_FILE', None)
    negative_csv_file = getenv('NEGATIVE_CSV_FILE', None)
    snippet_csv_file = getenv('SNIPPET_CSV_FILE', None)
    calculate_execution_time(method=prepare_data, label='execution time', positive_csv_file=positive_csv_file,
                             negative_csv_file=negative_csv_file, snippet_csv_file=snippet_csv_file)


if __name__ == '__main__':
    app()
