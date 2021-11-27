from typing import Iterable, Any


class NotFittedError(Exception):
    """Custon exception to signal that the class has not been fitted"""
    pass


class LabelEncoder:
    """The LabelEncoder allows for the encoding of categorical variables into
    sorted integers.

    Args:
        handle_unknown (bool): Whether or not to treat unknown labels with their own ID.
    """
    DEFAULT_UNK_VALUE = 0
    DEFAULT_VOCAB_START = 1

    def __init__(self, handle_unknown: bool = True):
        self.handle_unknown = handle_unknown

        self._fitted = False

    def fit(self, data: Iterable):
        """Fits the LabelEncoder to a set of data elements.

        Args:
            data: Set of data elements.

        Returns:
            A fitted LabelEncoder.
        """
        start_pos = LabelEncoder.DEFAULT_VOCAB_START if self.handle_unknown else 0

        self.value_to_id = {value: i for i, value in enumerate(set(data), start=start_pos)}
        self.id_to_value = {i: value for value, i in self.value_to_id.items()}

        self._fitted = True

        return self

    def __len__(self) -> int:
        return len(self.value_to_id) + int(self.handle_unknown)

    def transform(self, value: Any) -> int:
        """Transforms a `value` to its corresponding ID. If the value is unknown
        and `handle_unknown` is set to false, it will yield a KeyError.

        Note: If the LabelEncoder has not been previously fit, it will raise a NotFittedError.

        Args:
            value: Value to transform.

        Returns:
            ID corresponding to the original value.
        """
        if not self._fitted:
            raise NotFittedError

        if self.handle_unknown:
            return self.value_to_id.get(value, LabelEncoder.DEFAULT_UNK_VALUE)
        else:
            return self.value_to_id[value]

    def inverse_transform(self, i: int) -> int:
        """Transforms an ID to its corresponding value.

        Note: If the LabelEncoder has not been previously fit, it will raise a NotFittedError.

        Args:
            i: ID to transform.

        Returns:
            Original value corresponding to the ID.
        """
        if not self._fitted:
            raise NotFittedError

        return self.id_to_value[i]
