# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import dataset_woz2
import dataset_sim
import dataset_multiwoz21
import dataset_multiwoz21_legacy
import dataset_aux_task
import dataset_unified


class DataProcessor(object):
    dataset_name = ""
    class_types = []
    slot_list = []
    label_maps = {}

    def __init__(self, dataset_config):
        # Load dataset config file.
        with open(dataset_config, "r", encoding='utf-8') as f:
            raw_config = json.load(f)
        self.class_types = raw_config['class_types'] # Required
        self.slot_list = raw_config['slots'] if 'slots' in raw_config else []
        self.label_maps = raw_config['label_maps'] if 'label_maps' in raw_config else {}
        self.dataset_name = raw_config['dataset_name'] if 'dataset_name' in raw_config else ""
        # If not slot list is provided, generate from data.
        if len(self.slot_list) == 0:
            self.slot_list = self._get_slot_list()

    def _get_slot_list(self):
        raise NotImplementedError()

    def get_train_examples(self):
        raise NotImplementedError()

    def get_dev_examples(self):
        raise NotImplementedError()

    def get_test_examples(self):
        raise NotImplementedError()


class Woz2Processor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_train_en.json'),
                                            'train', self.slot_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_validate_en.json'),
                                            'dev', self.slot_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_test_en.json'),
                                            'test', self.slot_list, self.label_maps, **args)


class Multiwoz21Processor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'train_dials.json'),
                                                  'train', self.class_types, self.slot_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'val_dials.json'),
                                                  'dev', self.class_types, self.slot_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'test_dials.json'),
                                                  'test', self.class_types, self.slot_list, self.label_maps, **args)


class Multiwoz21LegacyProcessor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_multiwoz21_legacy.create_examples(os.path.join(data_dir, 'train_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_multiwoz21_legacy.create_examples(os.path.join(data_dir, 'val_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'dev', self.slot_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_multiwoz21_legacy.create_examples(os.path.join(data_dir, 'test_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'test', self.slot_list, self.label_maps, **args)


class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'train.json'),
                                           'train', self.slot_list, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'dev.json'),
                                           'dev', self.slot_list, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'test.json'),
                                           'test', self.slot_list, **args)


class UnifiedDatasetProcessor(DataProcessor):
    def _get_slot_list(self):
        return dataset_unified.get_slot_list(self.dataset_name)
        
    def get_train_examples(self, data_dir, args):
        return dataset_unified.create_examples('train', self.dataset_name, self.class_types,
                                               self.slot_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_unified.create_examples('validation', self.dataset_name, self.class_types,
                                               self.slot_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_unified.create_examples('test', self.dataset_name, self.class_types,
                                               self.slot_list, self.label_maps, **args)


class AuxTaskProcessor(object):
    def get_aux_task_examples(self, data_dir, data_name, max_seq_length):
        file_path = os.path.join(data_dir, '{}_train.json'.format(data_name))
        return dataset_aux_task.create_examples(file_path, max_seq_length)


PROCESSORS = {"woz2": Woz2Processor,
              "sim-m": SimProcessor,
              "sim-r": SimProcessor,
              "multiwoz21": Multiwoz21Processor,
              "multiwoz21_legacy": Multiwoz21LegacyProcessor,
              "unified": UnifiedDatasetProcessor,
              "aux_task": AuxTaskProcessor}
