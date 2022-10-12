import torch
import os
import wget
import tarfile
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torch.utils.data import Dataset, DataLoader
try:
    from torchtext.legacy.datasets.text_classification import DATASETS, _create_data_from_iterator, TextClassificationDataset, LABELS
except:
    from torchtext.datasets.text_classification import DATASETS, TextClassificationDataset, _create_data_from_iterator, LABELS
from torchtext.datasets import IMDB as IMDB_iterator, AG_NEWS as AG_NEWS_iterator
from collections import Counter, OrderedDict
from torchtext.vocab import vocab as vocab_fn
import tqdm


def IMDB(root, ngrams=1, vocab=None, include_unk=False):
    """ Defines IMDB datasets.

    The labels include:

        - 0 : Negative
        - 1 : Positive

    Create supervised learning dataset: AG_NEWS

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = IMDB(ngrams=3)

    """
    test_iter = IMDB_iterator(root, split='test')

    tokenizer = get_tokenizer("basic_english")

    # build vocabulary
    if vocab is None:
        train_iter = IMDB_iterator(root, split='train')
        counter = Counter()
        for (label, line) in tqdm.tqdm(train_iter):
            counter.update(tokenizer(line))
        vocab = vocab_fn(OrderedDict(counter), min_freq=1)
        vocab.set_default_index(PAD_TOKEN)

    train_iter = IMDB_iterator(root, split='train')

    label_dict = {'pos': 1, 'neg': 0}

    def wrap_iterator(iter):
        for cls, tokens in iter:
            yield label_dict[cls], ngrams_iterator(tokenizer(tokens), ngrams)

    train_data, train_labels = _create_data_from_iterator(
        vocab, wrap_iterator(train_iter), include_unk)
    test_data, test_labels = _create_data_from_iterator(
        vocab, wrap_iterator(test_iter), include_unk)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")

    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))


def AG_NEWS(root, ngrams=1, vocab=None, include_unk=False):
    """ Defines IMDB datasets.

    The labels include:

        - 0 : Negative
        - 1 : Positive

    Create supervised learning dataset: AG_NEWS

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = AG_NEWS(ngrams=3)

    """
    test_iter = AG_NEWS_iterator(root, split='test')

    tokenizer = get_tokenizer("basic_english")

    # build vocabulary
    if vocab is None:
        train_iter = AG_NEWS_iterator(root, split='train')
        counter = Counter()
        for (label, line) in tqdm.tqdm(train_iter):
            counter.update(tokenizer(line))
        vocab = Vocab(counter, min_freq=1)

    train_iter = AG_NEWS_iterator(root, split='train')

    def wrap_iterator(iter):
        for cls, tokens in iter:
            yield cls - 1, ngrams_iterator(tokenizer(tokens), ngrams)

    train_data, train_labels = _create_data_from_iterator(
        vocab, wrap_iterator(train_iter), include_unk)
    test_data, test_labels = _create_data_from_iterator(
        vocab, wrap_iterator(test_iter), include_unk)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")

    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))


DATASETS['IMDB'] = IMDB
DATASETS['AG_NEWS'] = AG_NEWS

MAX_SENTENCE_LEN = {
    'AG_NEWS': 75,
    'SogouNews': 800,
    'DBpedia': 75,
    'YelpReviewPolarity': 200,
    'YelpReviewFull': 200,
    'YahooAnswers': 150,
    'AmazonReviewPolarity': 100,
    'AmazonReviewFull': 100,
    'IMDB': 300
}
PAD_TOKEN = '<pad>'


def get_text_dataset(dataset, root):
    train_dataset, test_dataset = DATASETS[dataset](
        root=root, ngrams=1, include_unk=True)

    return train_dataset, test_dataset


class CollateProcessor:
    """
    Function to create batch for text datasets
    """

    def __init__(self, dataset, vocab):
        self._max_length = MAX_SENTENCE_LEN[dataset]
        self._vocab = vocab

    def __call__(self, batch):
        labels, sentences = zip(*batch)
        labels = torch.LongTensor(labels)
        processed_sentences = \
            torch.ones((len(sentences), self._max_length),
                       dtype=torch.long)*self._vocab.stoi[PAD_TOKEN]
        for i, s in enumerate(sentences):
            l = min(self._max_length, len(s))
            processed_sentences[i, :l] = torch.LongTensor(s[:l])
        return processed_sentences, labels
