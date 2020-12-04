import pytest
from typing import Dict
import os
from ml_gym.blueprints.component_factory import ComponentFactory, Injector
from outlier_detection.constructables.constructables import CustomDatasetRepositoryConstructable
from outlier_detection.blueprints.mlp_blueprint import MLPCollator
from ml_gym.io.config_parser import YAMLConfigLoader


class TestReutersIterator:

    @pytest.fixture
    def full_config(self) -> Dict:
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "reuters_full_config.yml")
        config = YAMLConfigLoader.load(config_path)
        return config

    @pytest.fixture
    def components(self, full_config) -> Dict:
        component_names = list(full_config.keys())
        injection_mapping = {"id_mlp_standard_collator": MLPCollator}
        injector = Injector(injection_mapping)
        component_factory = ComponentFactory(injector)
        component_factory.register_component_type("DATASET_REPOSITORY", "DEFAULT", CustomDatasetRepositoryConstructable)
        components = component_factory.build_components_from_config(full_config, component_names)
        return components

    def test_dataset_iterator_from_constructable(self, components):
        train_dataset_iterator = components["dataset_iterators"]["train"]
        test_dataset_iterator = components["dataset_iterators"]["test"]
        sample, target, tag = train_dataset_iterator[0]
        assert len(train_dataset_iterator) == 6577
        assert len(test_dataset_iterator) == 2583
        assert list(sample.shape) == [100] and isinstance(target, str)

    def test_filtered_labels_iterator_from_constructable(self, components: Dict, full_config):
        dataset_iterators = components["filtered_labels_iterator"]
        filtered_labels = full_config["filtered_labels_iterator"]["config"]["filtered_labels"]
        for _, iterator in dataset_iterators.items():
            assert all([sample[1] in filtered_labels for sample in iterator])

    def test_mapped_labels_iterator_from_constructable(self, components: Dict, full_config: Dict):
        dataset_iterators = components["mapped_labels_iterator"]
        previous_labels = [label for mapping in full_config["mapped_labels_iterator"]
                           ["config"]["mappings"] for label in mapping["previous_labels"]]
        new_labels = [mapping["new_label"] for mapping in full_config["mapped_labels_iterator"]["config"]["mappings"]]
        non_existing_labels = [label for label in previous_labels if label not in new_labels]
        for _, iterator in dataset_iterators.items():
            assert all([sample[1] not in non_existing_labels for sample in iterator])

    def test_splitted_dataset_iterator_from_constructable(self, components: Dict):
        dataset_iterators = components["splitted_dataset_iterators"]
        assert len(dataset_iterators["train"]) == 3832 and len(dataset_iterators["val"]) == 1643 and len(dataset_iterators["test"]) == 2172

    def test_full_pipeline(self, components: Dict):
        data_loaders = components["data_loaders"]
        assert sum([len(b.samples) for b in iter(data_loaders["train"])]) == 3832
        assert sum([len(b.samples) for b in iter(data_loaders["val"])]) == 1643
        assert sum([len(b.samples) for b in iter(data_loaders["test"])]) == 2172
