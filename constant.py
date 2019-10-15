from pathlib import Path
HOME = str(Path.home())


RAW_DATA_DIR = f'{HOME}/data/raw'
INPUTS_DIR = f'{RAW_DATA_DIR}/inputs'
LABEL_DIR = f'{RAW_DATA_DIR}/labels'
LABEL_ADJ_PATH = f'{RAW_DATA_DIR}/label_adjacent_graph.txt'
LABEL_DESCRIPTION_PATH = f'{RAW_DATA_DIR}/label_description.txt'

PROCESSED_DIR = f'{HOME}/data/processed'
TRAIN_DIR = f'{PROCESSED_DIR}/train_files'
DEV_DIR = f'{PROCESSED_DIR}/dev_files'
VOCAB_PATH = f'{PROCESSED_DIR}/vocab.txt'
