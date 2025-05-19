import os
import random
import yaml
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloading_preprocessing import get_dataloaders
from model import CombinedModel
from visualization import plot_training_curves
from utils import set_seed, setup_logging, save_checkpoint, load_checkpoint
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    exp_name = config['exp_name']
    dataset_name = config['dataset_name']
    seed = config['seed']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # directories to save the respective results
    dataset_dir = config['dataset_dir']
    log_dir = os.path.join(config['log_dir'], exp_name)
    checkpoint_dir = os.path.join(config['checkpoint_dir'], exp_name)
    tensorboard_dir = os.path.join(config['tensorboard_dir'], exp_name)
    plots_dir = os.path.join(config['plots_dir'], exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Logs
    logger = setup_logging(log_dir, exp_name, log_to_console=config['log_to_console'])
    logger.info(f"Starting experiment: {exp_name}")

    # TensorBoard
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name,
        dataset_dir,
        batch_size=config['batch_size'],
        input_size=config['input_size'],
        train_split=config['train_split']
    )
    logger.info(f"Dataset {dataset_name} loaded.")

    model = CombinedModel(
        num_classes=config['num_classes'],
        freeze=True,
        lora_rank=config['lora_rank'],
        lora_scale=config['lora_scale']
    ).to(device)
    logger.info("Model created.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device="cuda")


    # Training
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        model.train()
        running_loss, correct, total = 0, 0, 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for i, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
        
            optimizer.zero_grad()

            # Mixed precision forward pass (float16 + float32 for efficiency)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item()

            # tensorboard logs for evry epoch
            step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Accuracy', preds.eq(targets).float().mean().item(), step)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation for every epoch
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                _, preds = outputs.max(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += targets.size(0)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        writer.add_scalars('Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch)
        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)

        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save checkpoint
        if epoch % 2 == 0 or epoch == config['num_epochs']-1 :
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(checkpoint_dir, f'epoch_{epoch}.pt'),
                train_acc=train_acc,
                val_acc=val_acc,
                loss=val_loss
            )
            
        # save the best model checkkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if config['save_best_only']:
                save_checkpoint(
                    model, optimizer, epoch,
                    os.path.join(checkpoint_dir, 'best_model.pt'),
                    train_acc=train_acc,
                    val_acc=val_acc,
                    loss=val_loss
                )

    logger.info(f"Training finished. Best Val Accuracy: {best_val_acc:.4f}")

    # Plot
    plot_training_curves(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        save_path=os.path.join(plots_dir, 'training_loss_accuracy_curve.png')
    )

    writer.close()

if __name__ == "__main__":
    main()
