from typing import List, Tuple
import tempfile
import pytest
import os
from outlier_detection.blueprints.mlp_blueprint import MLPBluePrint
from outlier_detection.blueprints.ata_blueprint import ATABluePrint
from outlier_detection.blueprints.sae_blueprint import SAEBluePrint
from ml_gym.blueprints.blue_prints import BluePrint
from ml_gym import create_blueprints, create_gym, setup_logging_environment, stop_logging_environment
import glob
import json
from ml_gym.util.logger import QueuedLogging


def get_minimal_config(config_path: str, num_indices: int, minimal_iterator_name: str, highest_order_iterator_name: str, subsequent_component_name: str) -> str:

    iterator_view_definition = \
        f"""
minimal_iterators:
  component_type_key: ITERATOR_VIEW
  variant_key: DEFAULT
  requirements:
  - name: iterators
    component_name: {highest_order_iterator_name}
    subscription: [train, val, test]
  config:
    applicable_splits: [train, val, test]
    num_indices: {num_indices}
        """

    def read_config_file(config_path) -> List[str]:
        with open(config_path, "r") as fp:
            config_lines = fp.read().split("\n")
        return config_lines

    def get_insert_position(config_lines: List[str]):
        for i, line in enumerate(config_lines):
            if f"{subsequent_component_name}:" in line:
                break
        return i

    config_lines = read_config_file(config_path)
    insert_pos = get_insert_position(config_lines)
    config_lines = [line.replace(f"component_name: {highest_order_iterator_name}", f"component_name: {minimal_iterator_name}")
                    for line in config_lines]
    config_lines.insert(insert_pos, iterator_view_definition)

    return "\n".join(config_lines)


class TestBluePrint:
    config_path_key = "config_path_key"
    minimal_iterator_name_key = "minimal_iterator_name"
    subsequent_component_name_key = "subsequent_component_name"
    highest_order_iterator_name_key = "highest_order_iterator_name"
    blue_print_key = "blue_print_key"
    model_name_key = "model_name_key"

    model_configs = [
        # MLP
        {
            config_path_key: "./pytest/gs_test_configs/mlp/ARR.yml",
            minimal_iterator_name_key: "minimal_iterators",
            subsequent_component_name_key: "data_loaders",
            highest_order_iterator_name_key: "feature_encoded_iterators",
            blue_print_key: MLPBluePrint,
            model_name_key: "MLP"
        },
        # ATA
        {
            config_path_key: "./pytest/gs_test_configs/ata/ARR.yml",
            minimal_iterator_name_key: "minimal_iterators",
            subsequent_component_name_key: "data_loaders",
            highest_order_iterator_name_key: "feature_encoded_iterators",
            blue_print_key: ATABluePrint,
            model_name_key: "ATA"
        },
        # SAE
        {
            config_path_key: "./pytest/gs_test_configs/sae/ARR.yml",
            minimal_iterator_name_key: "minimal_iterators",
            subsequent_component_name_key: "data_loaders",
            highest_order_iterator_name_key: "feature_encoded_iterators",
            blue_print_key: SAEBluePrint,
            model_name_key: "SAE"
        }
    ]

    def validate_training_output(experiment_path: str, ):
        def validate_metrics(experiment_path: str):
            metrics_path = os.path.join(experiment_path, "metrics.json")
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metric_key = list(metrics.keys())[0]
            assert len(metrics[metric_key]) == 1

        validate_metrics(experiment_path)

        config_path = os.path.join(experiment_path, "metrics.json")
        errout_path = os.path.join(experiment_path, "errout.txt")
        stdout_path = os.path.join(experiment_path, "stdout.txt")
        model_path = os.path.join(experiment_path, "checkpoints/model_0.pt")
        optimizer_path = os.path.join(experiment_path, "checkpoints/optimizer_0.pt")
        state_path = os.path.join(experiment_path, "checkpoints/state_0.json")

        assert os.path.isfile(config_path)
        assert os.path.isfile(errout_path)
        assert os.path.isfile(stdout_path)
        assert os.path.isfile(model_path)
        assert os.path.isfile(optimizer_path)
        assert os.path.isfile(state_path)

    @pytest.fixture
    def config_stuff(self, request) -> Tuple[str, BluePrint]:
        config_path = request.param[TestBluePrint.config_path_key]
        minimal_iterator_name = request.param[TestBluePrint.minimal_iterator_name_key]
        subsequent_component_name = request.param[TestBluePrint.subsequent_component_name_key]
        highest_order_iterator_name = request.param[TestBluePrint.highest_order_iterator_name_key]
        blue_print_class = request.param[TestBluePrint.blue_print_key]
        model_name = request.param[TestBluePrint.model_name_key]

        num_indices = 30
        minimal_config = get_minimal_config(config_path, num_indices, minimal_iterator_name,
                                            highest_order_iterator_name, subsequent_component_name)
        tf = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        tf.write(minimal_config)
        tf.flush()
        tf.close()
        path = tf.name
        yield path, blue_print_class, model_name
        os.remove(path)

    @pytest.fixture
    def tmp_dashify_logging_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def tmp_general_logging_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.mark.parametrize("config_stuff", model_configs, indirect=True)
    def test_blue_print(self, tmp_dashify_logging_dir, tmp_general_logging_dir, config_stuff):
        num_epochs = 1
        run_mode = "TRAIN"
        dashify_logging_path = tmp_dashify_logging_dir
        gs_config_path, blue_print_class, model_name = config_stuff
        process_count = 1
        device_ids = None
        QueuedLogging._instance = None
        setup_logging_environment(tmp_general_logging_dir)
        gym = create_gym(process_count=process_count, device_ids=device_ids)
        blueprints = create_blueprints(blue_print_class=blue_print_class,
                                       run_mode=run_mode,
                                       gs_config_path=gs_config_path,
                                       dashify_logging_path=dashify_logging_path,
                                       num_epochs=num_epochs)
        gym.add_blue_prints(blueprints)
        gym.run(parallel=False)
        stop_logging_environment()
        experiment_path = os.path.join(glob.glob(os.path.join(tmp_dashify_logging_dir, "*"))[0], f"{model_name}/0")
        TestBluePrint.validate_training_output(experiment_path)
