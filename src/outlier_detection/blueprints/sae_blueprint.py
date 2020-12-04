import torch
from typing import Dict, List
from ml_gym.blueprints.blue_prints import BluePrint
from ml_gym.gym.jobs import AbstractGymJob, GymJob
from ml_gym.batch import DatasetBatch
from dataclasses import dataclass
from ml_gym.data_handling.postprocessors.collator import CollatorIF
from ml_gym.blueprints.component_factory import ComponentFactory, Injector
from outlier_detection.constructables.constructables import CustomDatasetRepositoryConstructable, CustomModelRegistryConstructable, \
    AtisFactoryConstructable, KddFactoryConstructable, LossFunctionRegistryConstructable


@dataclass
class SAECollator(CollatorIF):
    target_original_sample_publication_key: str
    target_class_publication_key: str

    def __call__(self, batch: List[torch.Tensor]) -> DatasetBatch:
        """ Takes a batch and collates into a proper TrainBatch.
        :param batch
        :return:
        """    # takes a batch and collates into a proper TrainBatch.
        inputs = [item[0] for item in batch]
        inputs = torch.stack(inputs)
        inputs = inputs.view(inputs.shape[0], -1).float()
        class_targets_tensor = torch.tensor([item[1] for item in batch]).to(inputs[0].device)
        tags = [item[2] for item in batch]  # to(inputs[0].device)

        target_partitions = {self.target_original_sample_publication_key: inputs.clone(),
                             self.target_class_publication_key: class_targets_tensor}
        return DatasetBatch(targets=target_partitions, tags=tags, samples=inputs)


class SAEBluePrint(BluePrint):
    def __init__(self, run_mode: AbstractGymJob.Mode, config: Dict, epochs: List[int], dashify_logging_dir: str, grid_search_id: str, run_id: str):
        model_name = "SAE"
        dataset_name = ""
        super().__init__(model_name, dataset_name, epochs, config, dashify_logging_dir, grid_search_id, run_id)
        self.run_mode = run_mode

    def construct(self) -> AbstractGymJob:
        experiment_info = self.get_experiment_info(self.dashify_logging_dir, self.grid_search_id, self.model_name, self.dataset_name, self.run_id)
        component_names = ["model", "trainer", "optimizer", "evaluator"]
        injection_mapping = {"id_sae_collator": SAECollator}
        injector = Injector(injection_mapping)
        component_factory = ComponentFactory(injector)
        component_factory.register_component_type("DATASET_REPOSITORY", "DEFAULT", CustomDatasetRepositoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "ATIS", AtisFactoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "KDD", KddFactoryConstructable)
        component_factory.register_component_type("LOSS_FUNCTION_REGISTRY", "DEFAULT", LossFunctionRegistryConstructable),
        component_factory.register_component_type("MODEL_REGISTRY", "DEFAULT", CustomModelRegistryConstructable)
        components = component_factory.build_components_from_config(self.config, component_names)
        gym_job = GymJob(self.run_mode, experiment_info=experiment_info, epochs=self.epochs, **components)
        return gym_job
