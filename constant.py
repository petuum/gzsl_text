# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
import os


# main directory for ICD code data, default set to $HOME/icd_data/
ICD_DATA_DIR = os.environ.get('ICD_DATA_DIR', os.path.join(str(Path.home()), 'icd_data'))
if not os.path.exists(ICD_DATA_DIR):
    raise ValueError(
        f"Cannot find main ICD coding data directory: {ICD_DATA_DIR}, "
        f"please setup the directory according to README.md"
    )


# processed MIMIC-III data
PROCESSED_DIR = f'{ICD_DATA_DIR}/processed'
if not os.path.exists(PROCESSED_DIR):
    os.mkdir(PROCESSED_DIR)

# path for caching all processed training data
CACHE_PATH = f'{PROCESSED_DIR}/preloaded_train.npz'

# resources including vocabs, embeddings, ICD code domain knowledge, keywords etc
RESOURCES_DIR = f'{ICD_DATA_DIR}/resources'
VOCAB_PATH = f'{RESOURCES_DIR}/vocab.txt'
VOCAB_DICT_PATH = f'{RESOURCES_DIR}/vocab_to_ix.pkl'
EMBEDDING_PATH = f'{RESOURCES_DIR}/icd_word_emb.pkl'
KEYWORDS_PATH = f'{RESOURCES_DIR}/mimic3_note_keywords.pkl'
ICD_CODE_HIERARCHY_PATH = f'{RESOURCES_DIR}/icd_code_hierarchy.txt'
ICD_CODE_DESC_DATA_PATH = f'{RESOURCES_DIR}/code_desc_vocab.npz'

for file in [VOCAB_DICT_PATH, EMBEDDING_PATH, KEYWORDS_PATH, ICD_CODE_HIERARCHY_PATH, ICD_CODE_DESC_DATA_PATH]:
    assert os.path.exists(file), f"{file} is missing," \
                                 f" please download and extract resources.tar.gz according to README.md"

SPLIT_DIR = f'{RESOURCES_DIR}/splits'
assert os.path.isdir(SPLIT_DIR), "Data splits is missing, " \
                                 "please download and extract resources.tar.gz according to README.md"

# directory to save models
MODEL_DIR = f'{ICD_DATA_DIR}/models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# directory to save features used in train_gan.py
FEATURE_DIR = f'{ICD_DATA_DIR}/features'
if not os.path.exists(FEATURE_DIR):
    os.mkdir(FEATURE_DIR)
