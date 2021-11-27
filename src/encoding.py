from typing import Iterable, Any


class NotFittedError(Exception):
    """Custon exception to signal that the class has not been fitted"""
    pass


class Vocabulary:
    DEFAULT_UNK_VALUE = 0
    DEFAULT_VOCAB_START = 1

    def __init__(self, handle_unknown: bool = True):
        self.handle_unknown = handle_unknown

        self._fitted = False

    def fit(self, data: Iterable):
        self.value_to_id = {value: i
                            for i, value in enumerate(set(data),
                                                      start=Vocabulary.DEFAULT_VOCAB_START)}

        self.id_to_value = {i: value for value, i in self.value_to_id.items()}

        self._fitted = True

        return self

    def __len__(self) -> int:
        return len(self.value_to_id) + int(self.handle_unknown)

    def transform(self, value: Any) -> int:
        if not self._fitted:
            raise NotFittedError

        if self.handle_unknown:
            return self.value_to_id.get(value, Vocabulary.DEFAULT_UNK_VALUE)
        else:
            return self.value_to_id[value]

    def inverse_transform(self, i: int) -> int:
        if not self._fitted:
            raise NotFittedError

        return self.id_to_value[i]
