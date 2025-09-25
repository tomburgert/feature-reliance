import sys
import argparse
from itertools import product
import subprocess

python_exe = sys.executable

parser = argparse.ArgumentParser(description="Run image classification experiment for a participant.")
parser.add_argument("-d", "--datasets", nargs="+", type=str, required=True)
parser.add_argument("-m", "--models", nargs="+", type=str, required=True)
parser.add_argument("--pretrained", action='store_true', help="Use pretrained model")
parser.add_argument("-l", "--exp_dir", type=str, default="/data/tomburgert/data/logs_feature_bias/computer_vision")
args = parser.parse_args()

print(args.datasets)

# Path to the base configuration file
CONFIG_FILE_PATH = 'conf/config.yaml'
TEST_SCRIPT_PATH = 'test.py'

# Define the parameters to iterate over
BASE_PARAMETERS = {
    'model.pretrained': [args.pretrained],
    'logging.exp_dir': [args.exp_dir],
    'model.pretrained_version': [0],
    'params.cuda_no': [5],
    'params.seed': [1]
}

base_test_transform = 'resize'

# SINGLE SUPPRESION

NO_SUPPRESSION_PARAMETERS = {
    'dataaug.test_augmentations': [base_test_transform],
    'params.protocol_name': ['{}_no_suppression'.format(base_test_transform)]
}

PATCH_SHUFFLE_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_patchshuffle'.format(base_test_transform)],
    'dataaug.grid_size' : [2, 4, 6, 8, 10, 12, 14, 16],
    'params.protocol_name': ['{}_patch_shuffle'.format(base_test_transform)]
}

PATCH_ROTATION_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_patchrotation'.format(base_test_transform)],
    'dataaug.grid_size' : [2, 4, 6, 8, 10, 12, 14, 16],
    'params.protocol_name': ['{}_patch_rotation'.format(base_test_transform)]
}

BILATERAL_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_bilateral'.format(base_test_transform)],
    'dataaug.bilateral_d': [4, 6, 8, 10, 12, 14],
    'dataaug.sigma_color': [50, 80, 110, 140, 170, 200],
    'dataaug.sigma_space': [75],
    'params.protocol_name': ['{}_bilateral'.format(base_test_transform)],
}

GAUSSIAN_BLUR_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_gaussianblur'.format(base_test_transform)],
    'dataaug.gaussian_k': [3, 5, 7, 9, 11, 13, 15],
    'dataaug.gaussian_sigma': [0.33, 0.66, 1.0, 1.33, 1.66, 2, 2.33],
    'params.protocol_name': ['{}_gaussianblur2'.format(base_test_transform)],
}


NLMEANS_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_nlmeans'.format(base_test_transform)],
    'dataaug.nlmeans_h': [5, 10, 15, 20, 25],
    'dataaug.template_window_size': [14],
    'dataaug.search_window_size': [21],
    'params.protocol_name': ['{}_nlmeans'.format(base_test_transform)]
}

GRAYSCALE_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_grayscale'.format(base_test_transform)],
    'dataaug.gray_alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
    'params.protocol_name': ['{}_grayscale'.format(base_test_transform)]
}

CHANNELSHUFFLE_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_channelshuffle'.format(base_test_transform)],
    'params.protocol_name': ['{}_channel_shuffle'.format(base_test_transform)]
}

# DOUBLE SUPPRESION

PATCH_SHUFFLE_AND_GRAYSCALE_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_patchshuffle_grayscale'.format(base_test_transform)],
    'dataaug.grid_size' : [2, 4, 6, 8, 10, 12, 14, 16],
    'dataaug.gray_alpha': [1.0],
    'params.protocol_name': ['{}_patch_shuffle_grayscale'.format(base_test_transform)]
}

BILATERAL_AND_PATCH_SHUFFLE_PARAMETERS1 = {
    'dataaug.test_augmentations': ['{}_bilateral_patchshuffle'.format(base_test_transform)],
    'dataaug.grid_size' : [2, 4, 6, 8, 10, 12],
    'dataaug.bilateral_d': [4, 6, 8, 10, 12, 14],
    'dataaug.sigma_color': [50, 80, 110, 140, 170, 200],
    'dataaug.sigma_space': [75],
    'params.protocol_name': ['{}_bilateral_patch_shuffle'.format(base_test_transform)]
}

BILATERAL_AND_PATCH_SHUFFLE_PARAMETERS2 = {
    'dataaug.test_augmentations': ['{}_bilateral_patchshuffle'.format(base_test_transform)],
    'dataaug.grid_size' : [2, 4, 6, 8, 10, 12, 14, 16],
    'dataaug.bilateral_d': [12],
    'dataaug.sigma_color': [170],
    'dataaug.sigma_space': [75],
    'params.protocol_name': ['{}_bilateral_patch_shuffle2'.format(base_test_transform)]
}

BILATERAL_AND_GRAYSCALE_PARAMETERS = {
    'dataaug.test_augmentations': ['{}_bilateral_grayscale'.format(base_test_transform)],
    'dataaug.bilateral_d': [4, 6, 8, 10, 12, 14],
    'dataaug.sigma_color': [50, 80, 110, 140, 170, 200],
    'dataaug.gray_alpha': [1.0],
    'dataaug.sigma_space': [75],
    'params.protocol_name': ['{}_bilateral_grayscale'.format(base_test_transform)]
}


ALL_SETUPS = [
    NO_SUPPRESSION_PARAMETERS,
    PATCH_SHUFFLE_PARAMETERS,
    PATCH_ROTATION_PARAMETERS,
    BILATERAL_PARAMETERS,
    GAUSSIAN_BLUR_PARAMETERS,
    NLMEANS_PARAMETERS,
    GRAYSCALE_PARAMETERS,
    CHANNELSHUFFLE_PARAMETERS,
    PATCH_SHUFFLE_AND_GRAYSCALE_PARAMETERS,
    BILATERAL_AND_PATCH_SHUFFLE_PARAMETERS1,
    BILATERAL_AND_PATCH_SHUFFLE_PARAMETERS2,
    BILATERAL_AND_GRAYSCALE_PARAMETERS
]

for dataset in args.datasets:
    for model in args.models:

        BASE_PARAMETERS.update(
            {
                'model.name': [model],
                'params.dataset': [dataset]
            }
        )

        for TRANSFORMATION_PARAMETERS in ALL_SETUPS:

            ALL_PARAMETERS = {**BASE_PARAMETERS, **TRANSFORMATION_PARAMETERS}

            # Special handling for bilateral
            if ALL_PARAMETERS['params.protocol_name'][0] in ['resize_bilateral', 'resize_bilateral_grayscale']:
                # Zip bilateral_d and sigma_color
                d_list = ALL_PARAMETERS.pop('dataaug.bilateral_d')
                c_list = ALL_PARAMETERS.pop('dataaug.sigma_color')
                
                zipped_pairs = list(zip(d_list, c_list))

                # Keep other parameter keys and values
                other_keys = list(ALL_PARAMETERS.keys())
                other_values = list(ALL_PARAMETERS.values())

                # Add placeholders for zipped keys
                other_keys += ['dataaug.bilateral_d', 'dataaug.sigma_color']

                # Create parameter combinations with zipped values
                for other_combination in product(*other_values):
                    for d, c in zipped_pairs:
                        param_updates = dict(zip(other_keys, list(other_combination) + [d, c]))
                        print(f"Running test with parameters: {param_updates}")
                        overrides = [f"{key}={value}" for key, value in param_updates.items()]
                        subprocess.run([python_exe, TEST_SCRIPT_PATH, *overrides])

            elif ALL_PARAMETERS['params.protocol_name'][0] == 'resize_bilateral_patch_shuffle':
                # Zip bilateral_d and sigma_color
                d_list = ALL_PARAMETERS.pop('dataaug.bilateral_d')
                c_list = ALL_PARAMETERS.pop('dataaug.sigma_color')
                g_list = ALL_PARAMETERS.pop('dataaug.grid_size')

                zipped_pairs = list(zip(d_list, c_list, g_list))

                # Keep other parameter keys and values
                other_keys = list(ALL_PARAMETERS.keys())
                other_values = list(ALL_PARAMETERS.values())

                # Add placeholders for zipped keys
                other_keys += ['dataaug.bilateral_d', 'dataaug.sigma_color', 'dataaug.grid_size']

                # Create parameter combinations with zipped values
                for other_combination in product(*other_values):
                    for d, c, g in zipped_pairs:
                        param_updates = dict(zip(other_keys, list(other_combination) + [d, c, g]))
                        print(f"Running test with parameters: {param_updates}")
                        overrides = [f"{key}={value}" for key, value in param_updates.items()]
                        subprocess.run([python_exe, TEST_SCRIPT_PATH, *overrides])

            elif ALL_PARAMETERS['params.protocol_name'][0] == 'resize_gaussianblur2':
                # Zip bilateral_d and sigma_color
                k_list = ALL_PARAMETERS.pop('dataaug.gaussian_k')
                s_list = ALL_PARAMETERS.pop('dataaug.gaussian_sigma')

                zipped_pairs = list(zip(k_list, s_list))

                # Keep other parameter keys and values
                other_keys = list(ALL_PARAMETERS.keys())
                other_values = list(ALL_PARAMETERS.values())

                # Add placeholders for zipped keys
                other_keys += ['dataaug.gaussian_k', 'dataaug.gaussian_sigma']

                # Create parameter combinations with zipped values
                for other_combination in product(*other_values):
                    for k, s in zipped_pairs:
                        param_updates = dict(zip(other_keys, list(other_combination) + [k, s]))
                        print(f"Running test with parameters: {param_updates}")
                        overrides = [f"{key}={value}" for key, value in param_updates.items()]
                        subprocess.run([python_exe, TEST_SCRIPT_PATH, *overrides])

            else:
                # Default Cartesian product for other transformations
                parameter_combinations = list(product(*ALL_PARAMETERS.values()))
                parameter_keys = list(ALL_PARAMETERS.keys())

                for combination in parameter_combinations:
                    param_updates = dict(zip(parameter_keys, combination))
                    print(f"Running test with parameters: {param_updates}")
                    overrides = [f"{key}={value}" for key, value in param_updates.items()]
                    subprocess.run([python_exe, TEST_SCRIPT_PATH, *overrides])
