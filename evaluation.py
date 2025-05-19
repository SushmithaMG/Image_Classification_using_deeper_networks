import os
import torch
import yaml
import argparse
from dataset import get_dataloaders
from model import CombinedModel
from utils import load_checkpoint, setup_logging, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file (.pt)')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    exp_name = config['exp_name']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    # Logs
    log_dir = os.path.join(config['log_dir'], exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, exp_name + "_test", log_to_console=True)

    # Load Data
    _, _, test_loader = get_dataloaders(
        config['dataset_name'],
        config['dataset_dir'],
        batch_size=config['batch_size'],
        input_size=config['input_size'],
        train_split=config['train_split']
    )
    logger.info(f"Test dataset {config['dataset_name']} loaded.")

    # Load model
    model = CombinedModel(
        num_classes=config['num_classes'],
        freeze=True,
        lora_rank=config['lora_rank'],
        lora_scale=config['lora_scale']
    ).to(device)

    model, optimizer, _ = load_checkpoint(model, optimizer, args.checkpoint, device=device)
    model.eval()

    # Test
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    test_accuracy = correct / total
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Test Accuracy: {test_accuracy:.4f} saved")

if __name__ == "__main__":
    main()
