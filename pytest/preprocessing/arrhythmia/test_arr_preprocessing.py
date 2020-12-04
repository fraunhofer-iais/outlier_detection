import pytest
from typing import Dict
import os
from ml_gym.blueprints.component_factory import ComponentFactory, Injector
from outlier_detection.constructables.constructables import CustomDatasetRepositoryConstructable
import numpy as np
from outlier_detection.blueprints.mlp_blueprint import MLPCollator
from ml_gym.io.config_parser import YAMLConfigLoader


class TestArrhythmiaIterator:

    @pytest.fixture
    def full_config(self) -> Dict:
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "arr_full_config.yml")
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
        dataset_iterator = components["dataset_iterators"]["full"]
        sample, target, tag = dataset_iterator[0]
        assert len(dataset_iterator) == 452 and list(sample.shape) == [279] and isinstance(target, int)

    def test_filtered_labels_iterator_from_constructable(self, components: Dict, full_config):
        dataset_iterators = components["filtered_labels_iterator"]
        filtered_labels = full_config["filtered_labels_iterator"]["config"]["filtered_labels"]
        for _, iterator in dataset_iterators.items():
            assert all([sample[iterator.dataset_meta.target_pos] in filtered_labels for sample in iterator])

    def test_mapped_labels_iterator_from_constructable(self, components: Dict, full_config: Dict):
        dataset_iterators = components["mapped_labels_iterator"]
        previous_labels = [label for mapping in full_config["mapped_labels_iterator"]
                           ["config"]["mappings"] for label in mapping["previous_labels"]]
        new_labels = [mapping["new_label"] for mapping in full_config["mapped_labels_iterator"]["config"]["mappings"]]
        non_existing_labels = [label for label in previous_labels if label not in new_labels]
        for _, iterator in dataset_iterators.items():
            assert all([sample[iterator.dataset_meta.target_pos] not in non_existing_labels for sample in iterator])

    def test_splitted_dataset_iterator_from_constructable(self, components: Dict):
        dataset_iterators = components["splitted_dataset_iterators"]
        assert len(dataset_iterators["train"]) == 186 and len(dataset_iterators["val"]) == 62 and len(dataset_iterators["test"]) == 63

    def test_combined_dataset_iterator_from_constructable(self, components: Dict):
        def compare_samples(s1, s2):
            return all([all([i == j for i, j in zip(s1[0], s2[0])]), s1[1] == s2[1], s1[2] == s2[2]])
        full_iterator = components["combined_dataset_iterators"]["full"]

        assert len(full_iterator) == 311
        lower = 0
        split_names = ["train", "val", "test"]
        for split_name in split_names:
            splitted_iterator = components["splitted_dataset_iterators"][split_name]
            assert compare_samples(splitted_iterator[0], full_iterator[lower])
            assert compare_samples(splitted_iterator[len(splitted_iterator)-1], full_iterator[lower + len(splitted_iterator) - 1])
            lower += len(splitted_iterator)

    def test_encoded_features_iterator_from_constructable(self, components: Dict):
        dataset_iterators = components["feature_encoded_iterators"]
        assert np.abs(np.mean([row[0][0] for row in dataset_iterators["train"]])) < 0.0000001  # check continuous scaler
        assert np.abs(np.mean([row[0][10] for row in dataset_iterators["train"]])) < 0.0000001  # check continuous scaler
        assert np.abs(np.mean([row[0][0] for row in dataset_iterators["test"]])) > 0.0000001  # check continuous scaler
        assert np.abs(np.mean([row[0][10] for row in dataset_iterators["test"]])) > 0.0000001  # check continuous scaler
        assert np.abs(np.mean([row[0][10] for row in dataset_iterators["val"]])) > 0.0000001  # check continuous scaler
        assert dataset_iterators["train"][4][0].shape[0] == 336
        assert dataset_iterators["val"][4][0].shape[0] == 336
        assert dataset_iterators["test"][4][0].shape[0] == 336

    def test_full_pipeline(self, components: Dict):
        data_loaders = components["data_loaders"]
        assert sum([len(b.samples) for b in iter(data_loaders["train"])]) == 186
        assert sum([len(b.samples) for b in iter(data_loaders["val"])]) == 62
        assert sum([len(b.samples) for b in iter(data_loaders["test"])]) == 63
