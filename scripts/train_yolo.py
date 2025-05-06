import argparse
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

def main(model_path, data_path, train_path):
    print(model_path, data_path, train_path)
    # Load a model from the given model configuration file
    model = YOLO(model_path)
    add_wandb_callback(model, enable_model_checkpointing=True)
    
    model.train(cfg=train_path, data=data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model using ultralytics")
    parser.add_argument(
        "--model_path",
        type=str,
        default="cfgs/yolo/model/hololens_combined_v11.yaml",
        help="Path to the model configuration file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="cfgs/yolo/data/hololens_combined.yaml",
        help="Path to the data configuration file"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="cfgs/yolo/train/hololens_combined_v11.yaml",
        help="Path to the training configuration file"
    )
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.train_path)
