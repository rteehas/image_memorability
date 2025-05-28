import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import argparse

class LaMemDataset(Dataset):
    """
    Custom Dataset for LaMem dataset.
    This script can be run with command-line arguments to specify data paths,
    hyperparameters, and model save location. Use --help for more information.
    """
    def __init__(self, csv_file, img_dir, transform=None, augment=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            augment (callable, optional): Optional augmentation to be applied
                on a sample.
        """
        self.memorability_scores_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.memorability_scores_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.memorability_scores_frame.iloc[idx, 0])
        # Ensure image filename has a common extension if not present, e.g. .jpg
        # This is a basic attempt, might need more robust handling
        if not os.path.splitext(img_name)[1]:
            img_name += '.jpg'

        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_name}, returning None. You might want to handle this more robustly.")
            # Return None or a placeholder; or raise an error
            # For now, let's assume we want to skip this sample if the image is not found,
            # which is not ideal for a real training loop but simplifies this step.
            # A better approach would be to filter out missing images during __init__
            # or handle them in the DataLoader's collate_fn.
            # For this exercise, we'll proceed and expect potential errors if not handled by caller.
            # However, to make it runnable, let's return a dummy tensor and a score of 0.
            # This would require the calling code to filter these out.
            # A more practical solution is to pre-filter the CSV.
            print(f"Error loading image: {img_name}. Please ensure the file exists.")
            # Returning a dummy tensor and score to avoid crashing the batch collation,
            # but this sample should ideally be skipped or handled.
            # This part of the code might need adjustment based on how missing files are handled.
            # For now, let's raise an error to highlight the issue.
            raise FileNotFoundError(f"Image file not found: {img_name}")


        memorability_score = self.memorability_scores_frame.iloc[idx, 1]
        # In the CSV, the filename is expected to be in a column named 'filename' (index 0)
        # and memorability score in a column named 'memorability' (index 1)
        # Adjust column names/indices if your CSV is different.
        # Example: self.memorability_scores_frame.loc[idx, 'filename']
        #          self.memorability_scores_frame.loc[idx, 'memorability_score']


        if self.augment:
            image = self.augment(image)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(memorability_score, dtype=torch.float)

# Define default transformations
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader # Already imported, but good to ensure

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

import torchvision.models as models
import torch.nn as nn

def get_memorability_model(device):
    """
    Loads a pre-trained ResNet50 model, freezes its convolutional layers,
    and modifies the classifier head to output a single continuous value.
    The model is then moved to the specified device.

    Args:
        device (torch.device): The device to move the model to ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The modified ResNet50 model.
    """
    # Load a pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Freeze convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Output a single continuous value

    # Move the model to the specified device
    model = model.to(device)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', model_save_path='best_memorability_model.pth'):
    """
    Trains the model and evaluates it on the validation set.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): The device to run training on.

    Returns:
        torch.nn.Module: The trained model (or best model based on validation).
    """
    best_spearman_corr = -1.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # Ensure labels are the correct shape for loss function, e.g., [batch_size, 1]
            loss = criterion(outputs, labels.unsqueeze(1) if labels.ndim == 1 else labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1) if labels.ndim == 1 else labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        spearman_corr = 0.0 # Default in case of issues
        if len(all_preds) > 1 and len(all_labels) > 1: # spearmanr needs at least 2 points
            try:
                spearman_corr, _ = spearmanr(all_preds, all_labels)
            except ValueError as e: # Catches issues like all predictions being the same
                print(f"Could not calculate Spearman correlation: {e}")
                spearman_corr = 0.0 # Or handle as NaN, depending on preference
        else:
            print("Not enough data points to calculate Spearman correlation.")


        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Spearman Corr: {spearman_corr:.4f}")

        if spearman_corr > best_spearman_corr:
            best_spearman_corr = spearman_corr
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with Spearman correlation: {spearman_corr:.4f}")

    print(f"Best Spearman Correlation: {best_spearman_corr:.4f}")
    return model

def load_model(model_path, device):
    """
    Loads a saved model state_dict into a new model instance.

    Args:
        model_path (str): Path to the saved model state_dict (.pth file).
        device (torch.device): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model, set to evaluation mode.
    """
    model = get_memorability_model(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths and Hyperparameters from args
    train_csv_path_actual = args.csv_train_file
    val_csv_path_actual = args.csv_val_file
    img_dir_actual = args.img_dir
    model_s_path = args.model_save_path
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    dummy_data_created = False
    dummy_train_csv_path = 'dummy_train_lamem_scores.csv'
    dummy_val_csv_path = 'dummy_val_lamem_scores.csv'
    dummy_img_dir = 'dummy_images_for_training/'
    all_dummy_filenames_local = []


    if args.use_dummy_data and \
       args.csv_train_file == 'train_annotations.csv' and \
       args.csv_val_file == 'val_annotations.csv' and \
       args.img_dir == 'images/' and \
       (not os.path.exists(args.csv_train_file) or \
        not os.path.exists(args.csv_val_file) or \
        not os.path.exists(args.img_dir)):
        
        print("Default paths used with --use_dummy_data flag, and data not found. Generating dummy data...")
        dummy_data_created = True
        train_csv_path_actual = dummy_train_csv_path
        val_csv_path_actual = dummy_val_csv_path
        img_dir_actual = dummy_img_dir

        # Create dummy data for testing
        dummy_train_data = {
            'filename': [f'train_img_{i}.jpg' for i in range(8)],
            'memorability': [0.8 + i*0.01 for i in range(8)] # Dummy scores
        }
        dummy_val_data = {
            'filename': [f'val_img_{i}.jpg' for i in range(4)],
            'memorability': [0.75 + i*0.02 for i in range(4)] # Dummy scores
        }

        df_train = pd.DataFrame(dummy_train_data)
        df_val = pd.DataFrame(dummy_val_data)

        df_train.to_csv(train_csv_path_actual, index=False)
        df_val.to_csv(val_csv_path_actual, index=False)

        if not os.path.exists(img_dir_actual):
            os.makedirs(img_dir_actual)

        all_dummy_filenames_local.extend(list(df_train['filename']))
        all_dummy_filenames_local.extend(list(df_val['filename']))

        for fname in all_dummy_filenames_local:
            try:
                if not os.path.exists(os.path.join(img_dir_actual, fname)):
                    Image.new('RGB', (224, 224), color='gray').save(os.path.join(img_dir_actual, fname))
            except ImportError:
                print("Pillow (PIL) is not installed. Cannot create dummy images.")
                break 
            except Exception as e:
                print(f"Error creating dummy image {fname}: {e}")
                break
        print("Dummy data and directories created.")

    # Instantiate datasets and dataloaders
    try:
        train_dataset = LaMemDataset(csv_file=train_csv_path_actual, img_dir=img_dir_actual, transform=data_transforms['train'])
        val_dataset = LaMemDataset(csv_file=val_csv_path_actual, img_dir=img_dir_actual, transform=data_transforms['val'])

        if not train_dataset or not val_dataset or len(train_dataset) == 0 or len(val_dataset) == 0 :
             raise ValueError("Dataset is empty. This might be due to issues with image creation or paths.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
        print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

        model = get_memorability_model(device)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        print("Model, optimizer, and criterion initialized.")
        print("Starting training...")

        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                                    num_epochs=num_epochs, device=device, model_save_path=model_s_path)
        
        print("Training completed.")
        print(f"Best model state_dict saved at: {model_s_path}")

        if os.path.exists(model_s_path):
            print(f"Attempting to load model from {model_s_path}...")
            try:
                loaded_model = load_model(model_s_path, device)
                print("Model loaded successfully.")
                dummy_input_tensor = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    prediction = loaded_model(dummy_input_tensor)
                print(f"Dummy prediction with loaded model: {prediction.item()}")
            except Exception as e:
                print(f"Error loading or testing the saved model: {e}")
        else:
            print(f"Model file {model_s_path} not found. Skipping load test.")

    except FileNotFoundError as e:
        print(f"Error during setup: {e}. Ensure data paths are correct and files exist.")
    except ValueError as e:
        print(f"ValueError during setup or training: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if dummy_data_created:
            print("Cleaning up dummy files...")
            if os.path.exists(dummy_train_csv_path):
                os.remove(dummy_train_csv_path)
            if os.path.exists(dummy_val_csv_path):
                os.remove(dummy_val_csv_path)
            
            if os.path.exists(dummy_img_dir):
                for fname in all_dummy_filenames_local:
                    fpath = os.path.join(dummy_img_dir, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                try:
                    os.rmdir(dummy_img_dir)
                except OSError as e:
                    print(f"Could not remove dummy directory {dummy_img_dir}: {e}")
        
        # Conditional cleanup of model, e.g., if it was part of a dummy run
        # For this task, let's assume we always clean up the model created by the script run if it exists
        # and if dummy data was used. If not using dummy data, user controls the file.
        if dummy_data_created and os.path.exists(model_s_path):
             os.remove(model_s_path)
             print(f"Cleaned up model file: {model_s_path}")
        elif not dummy_data_created and os.path.exists(model_s_path):
             print(f"Model file {model_s_path} was created with user-provided data paths and is not automatically cleaned up by this script run.")


        print("Cleanup finished or skipped for user data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an image memorability prediction model.')
    parser.add_argument('--csv_train_file', type=str, default='train_annotations.csv', required=False,
                        help='Path to training annotations CSV file. Required if not using --use_dummy_data with default paths.')
    parser.add_argument('--csv_val_file', type=str, default='val_annotations.csv', required=False,
                        help='Path to validation annotations CSV file. Required if not using --use_dummy_data with default paths.')
    parser.add_argument('--img_dir', type=str, default='images/', required=False,
                        help='Path to the directory containing all images. Required if not using --use_dummy_data with default paths.')
    parser.add_argument('--model_save_path', type=str, default='best_memorability_model.pth',
                        help='Path where the best model will be saved.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--use_dummy_data', action='store_true',
                        help='Use dummy data if default paths are specified and corresponding files/directory do not exist.')
    
    args = parser.parse_args()

    # Basic validation for required paths if not using dummy data with defaults
    if not args.use_dummy_data:
        if not os.path.exists(args.csv_train_file):
            parser.error(f"Training CSV file not found: {args.csv_train_file}. Please provide a valid path or use --use_dummy_data with default paths.")
        if not os.path.exists(args.csv_val_file):
            parser.error(f"Validation CSV file not found: {args.csv_val_file}. Please provide a valid path or use --use_dummy_data with default paths.")
        if not os.path.isdir(args.img_dir): # Check if it's a directory
            parser.error(f"Image directory not found or not a directory: {args.img_dir}. Please provide a valid path or use --use_dummy_data with default paths.")
    
    main(args)
