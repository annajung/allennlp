import pytest

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.hugging_face_datasets_reader import HuggingfaceDatasetSplitReader
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
import logging
logger = logging.getLogger(__name__)
from datasets import list_datasets
class HuggingFaceDataSetReaderTest:

    @pytest.mark.parametrize("dataset, config", (("afrikaans_ner_corpus", None), ("dbpedia_14", None), ("universal_dependencies", "af_afribooms"), ("trec", None), ("swahili", None)))
    def test_read_for_datasets(self, dataset, config):
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)




