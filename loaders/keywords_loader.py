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


import os
import pickle

from constant import KEYWORDS_PATH


def load_note_keywords_dictionary():
    with open(KEYWORDS_PATH, 'rb') as f:
        note_keywords = pickle.load(f)
    return note_keywords


def load_note_keywords(paths):
    note_keywords = load_note_keywords_dictionary()
    return [note_keywords[os.path.split(p)[-1]] for p in paths]
