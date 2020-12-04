import pytest
from typing import Dict
import os
from ml_gym.blueprints.component_factory import ComponentFactory, Injector
from outlier_detection.constructables.constructables import CustomDatasetRepositoryConstructable, KddFactoryConstructable
from outlier_detection.blueprints.mlp_blueprint import MLPCollator
import numpy as np
from ml_gym.io.config_parser import YAMLConfigLoader


def is_ci_deployment() -> bool:
    return os.getenv("od_deployment_type") == "CI"


class TestKddIterator:

    @pytest.fixture
    def full_config(self) -> Dict:
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "kdd_full_config.yml")
        config = YAMLConfigLoader.load(config_path)
        return config

    @pytest.fixture
    def components(self, full_config) -> Dict:
        component_names = list(full_config.keys())
        injection_mapping = {"id_mlp_standard_collator": MLPCollator}
        injector = Injector(injection_mapping)
        component_factory = ComponentFactory(injector)
        component_factory.register_component_type("DATASET_REPOSITORY", "DEFAULT", CustomDatasetRepositoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "KDD", KddFactoryConstructable)
        components = component_factory.build_components_from_config(full_config, component_names)
        return components

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_dataset_iterator_from_constructable(self, components):
        train_dataset_iterator = components["dataset_iterators"]["train"]
        test_dataset_iterator = components["dataset_iterators"]["test"]
        sample, target, tag = train_dataset_iterator[0]
        assert len(train_dataset_iterator) == 125973
        assert len(test_dataset_iterator) == 22544
        assert list(sample.shape) == [41] and isinstance(target, str)

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_mapped_to_main_attack_labels_iterator_from_constructable(self, components: Dict, full_config: Dict):
        dataset_iterators = components["mapped_to_main_attack_labels_iterator"]
        previous_labels = [label for mapping in full_config["mapped_to_main_attack_labels_iterator"]["config"]["mappings"]
                           for label in mapping["previous_labels"]]
        new_labels = [mapping["new_label"] for mapping in full_config["mapped_to_main_attack_labels_iterator"]["config"]["mappings"]]
        non_existing_labels = [label for label in previous_labels if label not in new_labels]
        for _, iterator in dataset_iterators.items():
            assert all([sample[iterator.dataset_meta.target_pos] not in non_existing_labels for sample in iterator])
            # assert all([sample[target_position] in new_labels for sample in iterator])

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_mapped_to_outlier_labels_iterator_from_constructable(self, components: Dict, full_config: Dict):
        dataset_iterators = components["mapped_to_outlier_labels_iterator"]
        previous_labels = [label for mapping in full_config["mapped_to_outlier_labels_iterator"]["config"]["mappings"]
                           for label in mapping["previous_labels"]]
        new_labels = [mapping["new_label"] for mapping in full_config["mapped_to_outlier_labels_iterator"]["config"]["mappings"]]
        non_existing_labels = [label for label in previous_labels if label not in new_labels]
        for _, iterator in dataset_iterators.items():
            assert all([sample[iterator.dataset_meta.target_pos] not in non_existing_labels for sample in iterator])
            # all previous labels have to be mapped to new labels (here: outlier / inlier)
            assert all([sample[iterator.dataset_meta.target_pos] in new_labels for sample in iterator])

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_combined_dataset_iterator_from_constructable(self, components: Dict):
        def compare_samples(s1, s2):
            return all([all([i == j for i, j in zip(s1[0], s2[0])]), s1[1] == s2[1], s1[2] == s2[2]])
        full_iterator = components["combined_dataset_iterators"]["full"]

        assert len(full_iterator) == 148517
        lower = 0
        for _, splitted_iterator in components["mapped_to_outlier_labels_iterator"].items():
            assert compare_samples(splitted_iterator[0], full_iterator[lower])
            assert compare_samples(splitted_iterator[len(splitted_iterator)-1], full_iterator[lower + len(splitted_iterator) - 1])
            lower += len(splitted_iterator)

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_encoded_features_iterator_from_constructable(self, components: Dict, full_config: Dict):
        dataset_iterators_raw = components["dataset_iterators"]
        categorical_feature_indices = full_config["feature_encoded_iterators"]["config"]["feature_encoding_configs"][0]["feature_names"]

        categorical_feature_values = np.array([sample_tensor[categorical_feature_indices]
                                               for _, iterator in dataset_iterators_raw.items()
                                               for sample_tensor, _, _ in iterator]).transpose()
        categorical_feature_lengths = np.sum([len(np.unique(feature_values)) for feature_values in list(categorical_feature_values)])

        encoded_feature_size = len(dataset_iterators_raw["train"][0][0]) - len(categorical_feature_indices) + categorical_feature_lengths

        dataset_iterators_encoded = components["feature_encoded_iterators"]
        for _, iterator in dataset_iterators_encoded.items():
            # check by encoding size, if categorical features are correctly encoded
            assert len(iterator[0][0]) == encoded_feature_size

        dataset_iterators_encoded = components["feature_encoded_iterators"]
        assert np.abs(np.mean([row[0][0] for row in dataset_iterators_encoded["train"]])) < 0.0000001  # check continuous scaler
        assert np.abs(np.mean([row[0][0] for row in dataset_iterators_encoded["test"]])) > 0.0000001  # check continuous scaler
        # test the shape of a sample
        assert dataset_iterators_encoded["train"][4][0].shape[0] == 126
        assert dataset_iterators_encoded["test"][4][0].shape[0] == 126

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_splitted_dataset_iterator_from_constructable(self, components: Dict):
        dataset_iterators = components["splitted_dataset_iterators"]
        assert len(dataset_iterators["train"]) == 88181 and len(
            dataset_iterators["val"]) == 37792 and len(dataset_iterators["test"]) == 22544

    @pytest.mark.skipif(is_ci_deployment(), reason="CI deployment")
    def test_full_pipeline(self, components: Dict):
        data_loaders = components["data_loaders"]
        assert sum([len(b.samples) for b in iter(data_loaders["train"])]) == 88181
        assert sum([len(b.samples) for b in iter(data_loaders["val"])]) == 37792
        assert sum([len(b.samples) for b in iter(data_loaders["test"])]) == 22544
