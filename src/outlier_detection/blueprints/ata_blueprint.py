import torch
from typing import Dict, List
from ml_gym.gym.trainer import TrainComponent
from outlier_detection.models.ata.train_component import ATATrainComponent
from outlier_detection.models.ata.eval_component import ATAEvalComponent
from ml_gym.gym.inference_component import InferenceComponent
from ml_gym.gym.evaluator import Evaluator
from ml_gym.blueprints.blue_prints import BluePrint
from ml_gym.gym.jobs import AbstractGymJob, GymJob
from sklearn.metrics import f1_score
from dataclasses import field, dataclass
from ml_gym.blueprints.constructables import ComponentConstructable
from ml_gym.registries.class_registry import ClassRegistry
from ml_gym.gym.post_processing import PredictPostProcessing
from ml_gym.data_handling.dataset_loader import DatasetLoader
from ml_gym.data_handling.postprocessors.collator import CollatorIF
from ml_gym.batch import DatasetBatch
from ml_gym.blueprints.component_factory import ComponentFactory, Injector
from outlier_detection.constructables.constructables import CustomDatasetRepositoryConstructable, CustomModelRegistryConstructable, \
    AtisFactoryConstructable, KddFactoryConstructable, LossFunctionRegistryConstructable


@dataclass
class ATATrainComponentConstructable(ComponentConstructable):
    loss_fun_config: Dict = field(default_factory=dict)
    post_processors_config: List[Dict] = field(default_factory=list)
    target_class_subscription_key: str = ""
    prediction_lp_loss_subscription_key: str = ""

    def _construct_impl(self) -> TrainComponent:
        def score_fun(y_true, y_pred):
            return 1 / f1_score(y_true, y_pred, average="binary", pos_label=1)

        prediction_post_processing_registry: ClassRegistry = self.get_requirement(
            "prediction_postprocessing_registry")
        loss_function_registry: ClassRegistry = self.get_requirement("loss_function_registry")
        train_loss_fun = loss_function_registry.get_instance(**self.loss_fun_config)
        postprocessors = [PredictPostProcessing(prediction_post_processing_registry.get_instance(**config))
                          for config in self.post_processors_config]
        inference_component = InferenceComponent(postprocessors, no_grad=False)
        train_component = ATATrainComponent(self.prediction_lp_loss_subscription_key, self.target_class_subscription_key,
                                            inference_component, train_loss_fun, score_fun)
        return train_component


@dataclass
class ATAEvalComponentConstructable(ComponentConstructable):
    train_split_name: str = ""
    metrics_config: List = field(default_factory=list)
    loss_funs_config: List = field(default_factory=list)
    post_processors_config: List[Dict] = field(default_factory=list)

    def _construct_impl(self) -> Evaluator:
        dataset_loaders: Dict[str, DatasetLoader] = self.get_requirement("data_loaders")
        loss_function_registry: ClassRegistry = self.get_requirement("loss_function_registry")
        metric_registry: ClassRegistry = self.get_requirement("metric_registry")
        prediction_post_processing_registry: ClassRegistry = self.get_requirement("prediction_postprocessing_registry")

        loss_funs = {conf["tag"]: loss_function_registry.get_instance(**conf) for conf in self.loss_funs_config}
        metric_funs = [metric_registry.get_instance(**conf) for conf in self.metrics_config]
        postprocessors = [PredictPostProcessing(prediction_post_processing_registry.get_instance(**config))
                          for config in self.post_processors_config]
        inference_component = InferenceComponent(postprocessors, no_grad=True)
        eval_component = ATAEvalComponent(inference_component, metric_funs, loss_funs, dataset_loaders, self.train_split_name)
        return eval_component


@dataclass
class ATACollator(CollatorIF):
    target_original_sample_publication_key: str
    target_class_publication_key: str

    def __call__(self, batch: List[torch.Tensor]):
        """ Takes a batch and collates into a proper TrainBatch.
        :param batch
        :return:
        """
        inputs = [item[0] for item in batch]
        inputs = torch.stack(inputs)
        inputs = inputs.view(inputs.shape[0], -1).float()
        original_sample_targets = inputs.clone()
        targets_class_tensor = torch.tensor(
            [item[1] for item in batch]).to(inputs[0].device)
        tags = [item[2] for item in batch]  # to(inputs[0].device)
        target_partitions = {self.target_original_sample_publication_key: original_sample_targets,
                             self.target_class_publication_key: targets_class_tensor}
        return DatasetBatch(targets=target_partitions, tags=tags, samples=inputs)


class ATABluePrint(BluePrint):
    def __init__(self, run_mode: AbstractGymJob.Mode, config: Dict, epochs: List[int], dashify_logging_dir: str, grid_search_id: str, run_id: str):
        model_name = "ATA"
        dataset_name = ""
        super().__init__(model_name, dataset_name, epochs, config, dashify_logging_dir, grid_search_id, run_id)
        self.run_mode = run_mode

    def construct(self) -> AbstractGymJob:
        experiment_info = self.get_experiment_info(self.dashify_logging_dir, self.grid_search_id,
                                                   self.model_name, self.dataset_name, self.run_id)
        component_names = ["model", "trainer", "optimizer", "evaluator"]
        injection_mapping = {"id_ata_collator": ATACollator}
        injector = Injector(injection_mapping)
        component_factory = ComponentFactory(injector)
        component_factory.register_component_type("DATASET_REPOSITORY", "DEFAULT", CustomDatasetRepositoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "ATIS", AtisFactoryConstructable)
        component_factory.register_component_type("DATASET_FACTORY", "KDD", KddFactoryConstructable)
        component_factory.register_component_type("LOSS_FUNCTION_REGISTRY", "DEFAULT", LossFunctionRegistryConstructable),
        component_factory.register_component_type("MODEL_REGISTRY", "DEFAULT", CustomModelRegistryConstructable)
        component_factory.register_component_type("TRAIN_COMPONENT", "ATA", ATATrainComponentConstructable),
        component_factory.register_component_type("EVAL_COMPONENT", "ATA", ATAEvalComponentConstructable)
        components = component_factory.build_components_from_config(self.config, component_names)
        gym_job = GymJob(self.run_mode, experiment_info=experiment_info, epochs=self.epochs, **components)
        return gym_job
