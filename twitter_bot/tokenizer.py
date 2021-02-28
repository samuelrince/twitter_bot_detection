from spacy.lang.en import English
from spacy.tokenizer import Tokenizer as _SpacyTokenizer
from typing import List, Optional, Union


class _Vocab:
    PAD = 0
    UNK = 1

    def __init__(self, max_vocab_size: Optional[int] = None) -> None:
        self.word2idx = dict()
        self.idx2word = []
        self._max_vocab_size = max_vocab_size

        # Add special tokens
        self.add_word('')           # Padding
        self.add_word('<unk>')      # Unknown token

    def add_word(self, word: str) -> None:
        if self._max_vocab_size:
            if self.__len__() >= self._max_vocab_size:
                return

        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def get_word_index(self, word) -> int:
        return self.word2idx.get(word.lower(), self.UNK)

    def get_index_word(self, index) -> str:
        if index < self.__len__():
            return self.idx2word[index]
        return '<unk>'

    def __len__(self) -> int:
        return len(self.idx2word)


class Tokenizer:

    def __init__(self,
                 max_vocab_size: Optional[int] = None,
                 max_sequence_length: Optional[int] = None) -> None:
        self.dictionary = _Vocab(max_vocab_size=max_vocab_size)
        self._max_seq_len = max_sequence_length

        # Spacy tokenizer
        nlp = English()
        self._spacy_tokenizer = _SpacyTokenizer(nlp.vocab)

    def build_vocab(self, corpus: List[str]) -> None:
        for item in corpus:
            for token in self._tokenize(item):
                self.dictionary.add_word(token)

    def tokenize(self, text: str) -> List[int]:
        res = list()
        for token in self._tokenize(text):
            res.append(self.dictionary.get_word_index(token))

        # Padding / truncating
        if self._max_seq_len:
            if len(res) < self._max_seq_len:
                while len(res) < self._max_seq_len:
                    res.append(self.dictionary.get_word_index(''))
            else:
                res = res[:self._max_seq_len]

        return res

    def _tokenize(self, text: str) -> List[str]:
        return [t.text for t in self._spacy_tokenizer(text.lower())]

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)
