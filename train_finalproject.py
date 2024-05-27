import os
import torch
import wandb
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Dict, List, Tuple
import numpy as np
from torchinfo import summary

def contains_nan(tensor):
    return tensor.isnan().any().item()

# Function to save the model and optimizer state
def save_checkpoint(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    wandb.save(filename)

# Training step function
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if contains_nan(X) or contains_nan(y):
            print("Skipping batch with NaNs")
            continue
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Log at each step (batch)
        wandb.log({
            "Epoch": epoch,
            "Batch": batch,
            "Train Loss": loss.item(),
            "Train Accuracy": (y_pred_class == y).sum().item() / len(y_pred),
            "Learning Rate": optimizer.param_groups[0]['lr']
        })

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

# Testing step function
def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, correct = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if contains_nan(X) or contains_nan(y):
                print("Skipping batch with NaNs")
                continue
            outputs = model(X)
            loss = loss_fn(outputs, y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    test_loss /= len(dataloader)
    test_acc = correct / len(dataloader.dataset)
    return test_loss, test_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Function to print metrics
def print_metrics(cm, cr, acc, lr):
    cm_df = pd.DataFrame(cm, index=["COVID", "Normal", "Pneumonia"], columns=["COVID", "Normal", "Pneumonia"])
    print("Confusion Matrix:\n", cm_df)
    cr_df = pd.DataFrame(cr).transpose()
    print("\nClassification Report:\n", cr_df)
    print(f"\nAccuracy Score: {acc:.4f}")
    print(f"Learning Rate: {lr:.6f}")

# Main training function
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          start_epoch: int,
          epochs: int,
          device: torch.device,
          config: Dict,
          scheduler=None) -> Dict[str, List]:

    # Set up wandb with API key
    wandb.login(key=config["wandb"]["api_key"])
    
    wandb.init(project=config["wandb"]["project_name"], config=config)
    sample_batch = next(iter(train_dataloader))
    input_size = tuple(sample_batch[0].shape)
    
    # Generate and log model summary
    model_summary = summary(model, 
                            input_size=input_size,
                            col_names=["input_size", "output_size", "num_params", "trainable"],
                            col_width=20,
                            row_settings=["var_names"])  # Use input size from dataloader
    wandb.log({"Model Summary": str(model_summary)})

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "accuracy_score": [],
        "confusion_matrix": [],
        "classification_report": []
    }
    
    model.to(device)

    for epoch in tqdm(range(start_epoch, epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           epoch=epoch)
        test_loss, test_acc, all_preds, all_labels, all_probs = test_step(model=model,
                                                                         dataloader=test_dataloader,
                                                                         loss_fn=loss_fn,
                                                                         device=device)

        if scheduler is not None:
            if np.isnan(train_loss):
                print(f"Train loss is NaN at epoch {epoch}, reducing learning rate.")
                scheduler.step(float('inf'))  # Trigger the scheduler as if the loss has stopped improving
            else:
                scheduler.step(train_loss)
        
        cm = confusion_matrix(all_labels, all_preds)
        cr = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        acc = accuracy_score(all_labels, all_preds)

        all_probs = np.nan_to_num(all_probs, nan=0.0)
        # Log at the end of each epoch
        wandb.log({
            "Epoch": epoch,
            "Train Loss (Epoch)": train_loss,
            "Train Accuracy (Epoch)": train_acc,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc,
            "Confusion Matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds, class_names=["COVID", "Normal", "Pneumonia"]),
            "Classification Report": cr,
            "Accuracy Score": acc,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Predicted Probs": all_probs  # Save predicted probabilities as a histogram
        })

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["accuracy_score"].append(acc)
        results["confusion_matrix"].append(cm)
        results["classification_report"].append(cr)

        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print_metrics(cm, cr, acc, optimizer.param_groups[0]['lr'])
        
        if epoch % config["training"]["save_interval"] == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(wandb.run.dir, f"{config['checkpoint']}_epoch_{epoch}.pth") # Save the checkpoint to wandb
            save_checkpoint(epoch, model, optimizer, checkpoint_path)
    
    wandb.finish()

    return results

