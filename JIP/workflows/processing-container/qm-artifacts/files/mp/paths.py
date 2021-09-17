# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
storage_path = os.environ['OPERATOR_IN_DIR'] #/home/simon/Schreibtisch/JIP Demo/' # <-- Change this path to were [.../]JIP/... lives
storage_data_path = os.path.join(storage_path, 'data')

# Original data paths. TODO: set necessary data paths.
original_data_paths = {'example_dataset_name': 'example_path'}

# Path that represents JIP data structure for training and inference
JIP_dir = os.path.join(storage_path, 'JIP')

# Login for Telegram Bot
telegram_login = {'chat_id': 'TODO', 'token': 'TODO'}
