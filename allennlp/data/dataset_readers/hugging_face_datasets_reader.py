from allennlp.data import DatasetReader, Instance, Field, Token
from allennlp.data.dataset_readers import DatasetReaderInput
from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List
import logging
import warnings

import torch.distributed as dist
from allennlp.data.fields import TextField, MultiLabelField, LabelField, SequenceLabelField, SequenceField

from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable
from datasets import load_dataset
from datasets.features import ClassLabel, Translation, Sequence
from datasets.features import Value

logger = logging.getLogger(__name__)
class HuggingfaceDatasetSplitReader(DatasetReader):

    def __init__(
            self,
            max_instances: Optional[int] = None,
            manual_distributed_sharding: bool = False,
            manual_multiprocess_sharding: bool = False,
            serialization_dir: Optional[str] = None,
            dataset_name: [str] = None,
            split: str = 'train',
            config_name: Optional[str] = None,
            builder: Optional[str] = None,
            data_files: Optional[dict] = None
    ) -> None:
        super().__init__(max_instances, manual_distributed_sharding, manual_multiprocess_sharding, serialization_dir)

        # It would be cleaner to create a separate reader object for different dataset
        self.dataset = None
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.index = -1

        if builder:
            assert data_files

        if builder:
            self.dataset = load_dataset(builder, data_files=data_files)
        else:
            if config_name:
                self.dataset = load_dataset(self.dataset_name, self.config_name, split=split)
            else:
                self.dataset = load_dataset(self.dataset_name, split=split)

    def _read(self, file_path) -> Iterable[Instance]:
        """
        Reads the instances from the given `file_path` and returns them as an
        `Iterable`.

        You are strongly encouraged to use a generator so that users can
        read a dataset in a lazy way, if they so choose.
        """
        self.index += 1
        return map(self.text_to_instance, self.dataset)

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        `Instance`.  The primary intended use for this is with a
        :class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        `DatasetReader` to process new text lets us accomplish this, as we can just call
        `DatasetReader.text_to_instance` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The `Predictor` will
        have to make some assumptions about the kind of `DatasetReader` that it's using, in order
        to pass it the right information.
        """
        # if it is Dataset then use feature
        features = self.dataset.features
        fields = dict()
        for feature in features:
            value = features[feature]

            if isinstance(value, ClassLabel):
                field = LabelField(inputs[0][feature], skip_indexing=True)

            elif isinstance(value, Value):
                if value.dtype == 'string':
                    field = TextField([Token(inputs[0][feature])])
                elif value.dtype == 'int32':
                    field = LabelField(inputs[0][feature], skip_indexing=True)
                else:
                    field = LabelField(inputs[0][feature], skip_indexing=True)

            elif isinstance(value, Translation):
                if value.dtype == 'dict':
                    # todo
                    print("todo")

            elif isinstance(value, Sequence):
                if value.dtype == 'list':
                    if value.feature.dtype == 'string':
                        # todo
                        print("todo")
                        field = SequenceLabelField(inputs[0][feature])

            fields[feature] = field

        return Instance(fields)




    def apply_token_indexers(self, instance: Instance) -> None:
        """
        If `Instance`s created by this reader contain `TextField`s without `token_indexers`,
        this method can be overriden to set the `token_indexers` of those fields.

        E.g. if you have you have `"source"` `TextField`, you could implement this method like this:

        ```python
        def apply_token_indexers(self, instance: Instance) -> None:
            instance["source"].token_indexers = self._token_indexers
        ```

        If your `TextField`s are wrapped in a `ListField`, you can access them via `field_list`.
        E.g. if you had a `"source"` field of `ListField[TextField]` objects, you could:

        ```python
        for text_field in instance["source"].field_list:
            text_field.token_indexers = self._token_indexers
        ```
        """
        pass
