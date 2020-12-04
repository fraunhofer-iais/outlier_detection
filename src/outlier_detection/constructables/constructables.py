from data_stack.repository.repository import DatasetRepository
from data_stack.io.storage_connectors import StorageConnectorFactory
from ml_gym.blueprints.constructables import DatasetRepositoryConstructable, ModelRegistryConstructable, ComponentConstructable, \
    LossFunctionRegistryConstructable
from outlier_hub.datasets.arrhythmia.factory import ArrhythmiaFactory
from dataclasses import dataclass
from outlier_detection.models.mlp.model import MLPClassifier
from outlier_detection.models.ata.model import AdversariallyTrainedAutoEncoder
from outlier_detection.models.sae.model import SupervisedAutoEncoder
from outlier_hub.datasets.reuters.factory import ReutersFactory
from outlier_hub.datasets.trec.factory import TrecFactory
from outlier_hub.datasets.atis.factory import AtisFactory
from outlier_hub.datasets.kdd.factory import KDDFactory
from data_stack.dataset.factory import BaseDatasetFactory
from outlier_detection.loss_functions.ata_loss_functions import AdvReconstructionBinnedLoss, AdvReconstructionThresholdedLoss
from outlier_detection.loss_functions.sae_loss_functions import SAELoss
from ml_gym.loss_functions.loss_functions import Loss


# =================== DATASET CONSTRUCTABLES ===================

@dataclass
class KddFactoryConstructable(ComponentConstructable):
    storage_connector_path: str = ""
    train_set_path: str = ""
    test_set_path: str = ""
    attack_type_mapping_path: str = ""
    feature_and_target_names_path: str = ""

    def _construct_impl(self) -> BaseDatasetFactory:
        storage_connector = StorageConnectorFactory.get_file_storage_connector(self.storage_connector_path)
        return KDDFactory(storage_connector=storage_connector, train_set_path=self.train_set_path,
                          test_set_path=self.test_set_path, attack_type_mapping_path=self.attack_type_mapping_path,
                          feature_and_target_names_path=self.feature_and_target_names_path)


@dataclass
class AtisFactoryConstructable(ComponentConstructable):
    train_set_path: str = ""
    val_set_path: str = ""
    test_set_path: str = ""
    storage_connector_path: str = ""

    def _construct_impl(self) -> BaseDatasetFactory:
        storage_connector = StorageConnectorFactory.get_file_storage_connector(self.storage_connector_path)
        return AtisFactory(storage_connector=storage_connector, train_set_path=self.train_set_path,
                           val_set_path=self.val_set_path, test_set_path=self.test_set_path)


@dataclass
class CustomDatasetRepositoryConstructable(DatasetRepositoryConstructable):

    def _construct_impl(self) -> DatasetRepository:
        dataset_repository = super()._construct_impl()
        storage_connector = StorageConnectorFactory.get_file_storage_connector(self.storage_connector_path)
        dataset_repository.register("arr", ArrhythmiaFactory(storage_connector))
        dataset_repository.register("reuters", ReutersFactory(storage_connector))
        dataset_repository.register("trec", TrecFactory(storage_connector))

        if self.has_requirement("atis_factory"):  # TODO this is a little bit of a hack ...
            atis_factory = self.get_requirement("atis_factory")
            dataset_repository.register("atis", atis_factory)

        if self.has_requirement("kdd_factory"):
            kdd_factory = self.get_requirement("kdd_factory")
            dataset_repository.register("kdd", kdd_factory)

        return dataset_repository


# =================== MODEL CONSTRUCTABLES ===================

@dataclass
class CustomModelRegistryConstructable(ModelRegistryConstructable):
    def _construct_impl(self):
        super()._construct_impl()
        self.model_registry.add_class("mlp", MLPClassifier)
        self.model_registry.add_class("ata", AdversariallyTrainedAutoEncoder)
        self.model_registry.add_class("sae", SupervisedAutoEncoder)
        return self.model_registry


# =================== LOSS CONSTRUCTABLES ===================


@dataclass
class LossFunctionRegistryConstructable(LossFunctionRegistryConstructable):
    class LossKeys:
        AdvReconstructionBinnedLoss = "AdvReconstructionBinnedLoss"
        AdvReconstructionThresholdedLoss = "AdvReconstructionThresholdedLoss"
        SAELoss = "SAELoss"

    def _construct_impl(self):
        loss_fun_registry = super()._construct_impl()
        default_mapping: [str, Loss] = {
            self.LossKeys.AdvReconstructionBinnedLoss: AdvReconstructionBinnedLoss,
            self.LossKeys.AdvReconstructionThresholdedLoss: AdvReconstructionThresholdedLoss,
            self.LossKeys.SAELoss: SAELoss
        }

        for key, loss_type in default_mapping.items():
            loss_fun_registry.add_class(key, loss_type)

        return loss_fun_registry
