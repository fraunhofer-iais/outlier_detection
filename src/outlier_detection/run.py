import argparse
from blueprints.ata_blueprint import ATABluePrint
from blueprints.mlp_blueprint import MLPBluePrint
from blueprints.sae_blueprint import SAEBluePrint
from enum import Enum
from ml_gym import create_blueprints, create_gym, setup_logging_environment, stop_logging_environment


class BlueprintTypes(Enum):
    ATA = ATABluePrint
    MLP = MLPBluePrint
    SAE = SAEBluePrint


def parse_args():
    parser = argparse.ArgumentParser(description='Run a grid search on CPUs or distributed over multiple GPUs')
    parser.add_argument('--model_type', choices=['MLP', 'SAE', 'ATA'], required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('--run_mode', choices=['TRAIN', 'EVAL'], required=True)
    parser.add_argument('--process_count', type=int, required=True, help='Max. number of processes running at a time.')
    parser.add_argument('--dashify_logging_path', type=str, required=True, help='Path to the dashify root logging directory')
    parser.add_argument('--text_logging_path', type=str, required=True, help='Path to python textual logging directory')
    parser.add_argument('--gs_config_path', type=str, required=True, help='Path to the grid search config')
    parser.add_argument('--gpus', type=int, nargs='+', help='Indices of GPUs to distribute the GS over', default=None)

    args = parser.parse_args()
    model_type = args.model_type
    num_epochs = args.num_epochs
    run_mode = args.run_mode
    dashify_logging_path = args.dashify_logging_path
    gs_config_path = args.gs_config_path
    process_count = args.process_count
    gpus = args.gpus
    text_logging_path = args.text_logging_path
    return model_type, num_epochs, run_mode, dashify_logging_path, text_logging_path, gs_config_path, process_count, gpus


if __name__ == '__main__':
    model_type, num_epochs, run_mode, dashify_logging_path, text_logging_path, gs_config_path, process_count, gpus = parse_args()
    setup_logging_environment(text_logging_path)
    gym = create_gym(process_count=process_count, device_ids=gpus)
    blue_print_type = BlueprintTypes[model_type].value
    blueprints = create_blueprints(blue_print_class=blue_print_type,
                                   run_mode=run_mode,
                                   gs_config_path=gs_config_path,
                                   dashify_logging_path=dashify_logging_path,
                                   num_epochs=num_epochs)
    gym.add_blue_prints(blueprints)
    gym.run(parallel=True)
    stop_logging_environment()
