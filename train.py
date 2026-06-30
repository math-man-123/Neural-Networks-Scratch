# pyright: reportMissingImports=false, reportUndefinedVariable=false

import json
from trainer import Trainer


# load training configs from json file
with open("train_configs.json", "r", encoding="utf-8") as file:
    train_configs = json.load(file)

# train new network with config
trainer = Trainer(**train_configs)
trainer.train()
