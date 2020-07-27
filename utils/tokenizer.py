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


import re


class Tokenizer(object):
    def __init__(self, vocab: dict):
        super(Tokenizer, self).__init__()
        self.vocab = vocab
        self.special_rep = 'NOUN'
        self.unk = "<UNK>"
        self.num = "<NUM>"
        self.pat = re.compile(r'(\[\*\*[^\[]*\*\*\])')

    def remove_special_token(self, sent):
        return self.pat.sub(self.special_rep, sent)

    @staticmethod
    def tokenize(sent):
        words = [s for s in re.split(r"\W+", sent) if s and not s.isspace()]
        return words

    def replace_unknowns_nums(self, words):
        tokens = []
        for word in words:
            if self.special_rep.lower() == word:
                continue

            if word in self.vocab:
                tokens.append(word)
            else:
                token = self.distinguish_unk_num(word)
                if len(tokens) == 0 or tokens[-1] != token:
                    tokens.append(token)
        return tokens

    def distinguish_unk_num(self, word):
        if self.only_numerals(word):
            return self.num
        else:
            return self.unk

    @staticmethod
    def only_numerals(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def process(self, input_text: str):
        input_text = self.remove_special_token(input_text)
        words_tokenized = self.tokenize(input_text)
        words = [word.lower().strip() for word in words_tokenized]
        words = self.replace_unknowns_nums(words)
        return words
