# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
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

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="Task database.")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "data.json")) as f:
        data = json.load(f)

    val_list_file = os.path.join(args.data_dir, "valListFile.json")
    if not os.path.isfile(val_list_file):
        val_list_file = os.path.join(args.data_dir, "valListFile.txt")
    with open(val_list_file) as f:
        val_set = f.read().splitlines()

    test_list_file = os.path.join(args.data_dir, "testListFile.json")
    if not os.path.isfile(test_list_file):
        test_list_file = os.path.join(args.data_dir, "testListFile.txt")
    with open(test_list_file) as f:
        test_set = f.read().splitlines()

    val = {}
    train = {}
    test = {}

    for k, v in data.items():
        if k in val_set:
            val[k] = v
        elif k in test_set:
            test[k] = v
        else:
            train[k] = v

    print(len(data), len(train), len(val), len(test))

    with open(os.path.join(args.data_dir, "train_dials.json"), "w+") as f:
        f.write(json.dumps(train, indent = 4))

    with open(os.path.join(args.data_dir, "val_dials.json"), "w+") as f:
        f.write(json.dumps(val, indent = 4))

    with open(os.path.join(args.data_dir, "test_dials.json"), "w+") as f:
        f.write(json.dumps(test, indent = 4))

if __name__ == "__main__":
    main()
