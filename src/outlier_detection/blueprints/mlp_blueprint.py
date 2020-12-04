from typing import Dict, List
from ml_gym.blueprints.blue_prints import BluePrint
from ml_gym.gym.jobs import AbstractGymJob, GymJob
from ml_gym.blueprints.component_factory import ComponentFactory, Injector
import torch
from ml_gym.batch import DatasetBatch
from ml_gym.data_handling.postprocessors.collator import CollatorIF
from dataclasses import dataclass
from outlier_detection.constructables.constructables import CustomDatasetRepositoryConstructable, CustomModelRegistryConstructable, \
    AtisFactoryConstructable, KddFactoryConstructable


@dataclass
class MLPCollator(CollatorIF):
    target_publication_key: str

    def __call__(self, batch: List[torch.Tensor]) -> DatasetBatch:
        """ Takes a batch and collates into a proper TrainBatch.
        :param batch
        :return:
        """
        inputs = [item[0] for item in batch]
        inputs = torch.stack(inputs)
        inputs = inputs.view(inputs.shape[0], -1).float()
        targets_tensor = torch.tensor([item[1] for item in batch]).to(inputs[0].device)
        tags = [item[2] for item in batch]  # to(inputs[0].device)
        target_partitions = {self.target_publication_key: targets_tensor}
        return DatasetBatch(targets=target_partitions, tags=tags, samples=inputs)


class MLPBluePrint(BluePrint):
    def __init__(self, run_mode: AbstractGymJob.Mode, config: Dict, epochs: List[int], dashify_logging_dir: str, grid_search_id: str, run_id: str):
        model_name = "MLP"
        dataset_name = ""
        super().__init__(model_name, dataset_name, epochs, config, dashify_logging_dir, grid_search_id, run_id)
        self.run_mode = run_mode

    def construct(self) -> AbstractGymJob:
        experiment_info = self.get_experiment_info(self.dashify_logging_dir, self.grid_search_id, self.model_name, self.dataset_name, self.run_id)
        component_names = ["model", "trainer", "optimizer", "evaluator"]
        injection_mapping = {"id_mlp_standard_collator": MLPCollator}
        injector = Injector(injection_mapping)
        component_factory = ComponentFactory(injector)
        component_factory.register_component_type("DATASET_REPOSITORY", "DEFAULT", CustomDatasetRepositoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "ATIS", AtisFactoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "KDD", KddFactoryConstructable)

        component_factory.register_component_type("MODEL_REGISTRY", "DEFAULT", CustomModelRegistryConstructable)
        components = component_factory.build_components_from_config(self.config, component_names)
        gym_job = GymJob(self.run_mode, experiment_info=experiment_info, epochs=self.epochs, **components)
        return gym_job
