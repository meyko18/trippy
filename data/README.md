## Supported datasets

Datasets should go into the ```data/``` folder.

### sim-M & sim-R:

```
git clone https://github.com/google-research-datasets/simulated-dialogue.git
```

### WOZ 2.0

The original URL (http://mi.eng.cam.ac.uk/~nm480/woz_2.0.zip) is not active anymore.

We provide the dataset in ```data/woz2```.

### MultiWOZ 2.0, 2.1 & 2.2

```
git clone https://github.com/budzianowski/multiwoz.git
unzip multiwoz/data/MultiWOZ_2.0.zip -d multiwoz/data/
unzip multiwoz/data/MultiWOZ_2.1.zip -d multiwoz/data/
mv multiwoz/data/MULTIWOZ2\ 2/ multiwoz/data/MultiWOZ_2.0
python3 multiwoz/data/MultiWOZ_2.2/convert_to_multiwoz_format.py --multiwoz21_data_dir=multiwoz/data/MultiWOZ_2.1 --output_file=multiwoz/data/MultiWOZ_2.2/data.json
cp multiwoz/data/MultiWOZ_2.1/valListFile.txt multiwoz/data/MultiWOZ_2.2/
cp multiwoz/data/MultiWOZ_2.1/testListFile.txt multiwoz/data/MultiWOZ_2.2/
python split_multiwoz_data.py --data_dir multiwoz/data/MultiWOZ_2.0
python split_multiwoz_data.py --data_dir multiwoz/data/MultiWOZ_2.1
python split_multiwoz_data.py --data_dir multiwoz/data/MultiWOZ_2.2
```

### MultiWOZ 2.1 legacy version

With "legacy version" we refer to the mid 2019 version of MultiWOZ 2.1, which can be found at https://doi.org/10.17863/CAM.41572

We used this version when we built TripPy. We provide the exact data that we used in ```data/MULTIWOZ2.1_legacy```.

The dataset has since been updated and the most recent version of MultiWOZ 2.1 differs slightly from the version we used for the experiments that we report in [TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://www.aclweb.org/anthology/2020.sigdial-1.4/). Our code supports both the new version as well as the legacy version of MultiWOZ.

### MultiWOZ 2.3

```
git clone https://github.com/lexmen318/MultiWOZ-coref.git
```

### MultiWOZ 2.4

```
git clone https://github.com/smartyfh/MultiWOZ2.4.git
```
