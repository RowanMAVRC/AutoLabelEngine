# Imports
from ultralytics import YOLO

# Settings
model_path = "cfgs/yolo/model/hololens_combined_v11.yaml"
data_path = "cfgs/yolo/data/hololens_combined.yaml"
train_path = "cfgs/yolo/train/hololend_combined_v11.yaml"

# Load a model
model = YOLO(model_path)  # build a new model from scratch

# Use the model
model.train(cfg=train_path, data=data_path)  # train the model