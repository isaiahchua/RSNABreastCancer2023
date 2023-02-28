#! /bin/bash

python traindensenet_ddp.py -c config/config.yaml
python traindensenet_ddp.py -c config/config2.yaml
python traindensenet_ddp.py -c config/config3.yaml
