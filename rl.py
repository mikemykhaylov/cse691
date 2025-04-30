import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
DATA_FILE = 'heuristic_moves.csv'
BATCH_SIZE = 128 # Adjust based on GPU memory
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5 # L2 regularization
EPOCHS = 50 # Adjust as needed
VAL_SPLIT = 0.1 # 10% for validation
SEED = 42 # For reproducibility

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# For reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Data Loading and Preprocessing ---

print("Loading data...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please place it in the same directory.")
    # Create a dummy file for testing structure if needed
    print("Creating a dummy data file for structure testing...")
    num_samples = 1000
    columns = ['episode', 'step']
    columns.extend([f'small_boards_{i}' for i in range(27**2)])
    columns.extend([f'large_board_{i}' for i in range(27)])
    columns.extend(['current_player', 'next_large_cell'])
    columns.extend([f'action_mask_{i}' for i in range(27**2)])
    columns.extend(['calculated_action'])
    dummy_data = np.random.randint(-1, 2, size=(num_samples, len(columns)-1)) # Random board states (-1, 0, 1)
    dummy_data = np.concatenate([np.arange(num_samples).reshape(-1,1), np.zeros((num_samples,1))], axis=1) # episode, step
    dummy_data = np.concatenate([dummy_data, np.random.randint(-1, 2, size=(num_samples, 729))], axis=1) # small_boards
    dummy_data = np.concatenate([dummy_data, np.random.randint(-1, 2, size=(num_samples, 27))], axis=1) # large_board
    dummy_data = np.concatenate([dummy_data, np.random.randint(0, 2, size=(num_samples, 1))], axis=1) # current_player (0 or 1)
    dummy_data = np.concatenate([dummy_data, np.random.randint(-1, 27, size=(num_samples, 1))], axis=1) # next_large_cell (-1 indicates anywhere)
    dummy_mask = np.random.randint(0, 2, size=(num_samples, 729)) # action_mask
    # Ensure at least one legal move for dummy data
    for i in range(num_samples):
        if dummy_mask[i].sum() == 0:
            dummy_mask[i, np.random.randint(0, 729)] = 1
    dummy_data = np.concatenate([dummy_data, dummy_mask], axis=1)
    # Generate valid calculated actions based on mask
    dummy_actions = []
    for i in range(num_samples):
        legal_indices = np.where(dummy_mask[i] == 1)[0]
        dummy_actions.append(np.random.choice(legal_indices))
    dummy_data = np.concatenate([dummy_data, np.array(dummy_actions).reshape(-1, 1)], axis=1) # calculated_action

    df = pd.DataFrame(dummy_data, columns=columns)
    df.to_csv(DATA_FILE, index=False)
    print(f"Dummy file '{DATA_FILE}' created with {num_samples} samples.")

# Define column names programmatically to avoid errors
small_board_cols = [f'small_boards_{i}' for i in range(729)]
large_board_cols = [f'large_board_{i}' for i in range(27)]
scalar_cols = ['current_player', 'next_large_cell']
action_mask_cols = [f'action_mask_{i}' for i in range(729)]
target_col = 'calculated_action'

# Extract data parts
small_boards_data = df[small_board_cols].values
large_board_data = df[large_board_cols].values
scalar_data = df[scalar_cols].values
action_mask_data = df[action_mask_cols].values
target_data = df[target_col].values

print("Data shapes:")
print("Small boards:", small_boards_data.shape)
print("Large board:", large_board_data.shape)
print("Scalars:", scalar_data.shape)
print("Action mask:", action_mask_data.shape)
print("Target:", target_data.shape)

# --- 2. Create PyTorch Dataset ---

class TicTacToe3DDataset(Dataset):
    def __init__(self, small_boards, large_board, scalars, masks, targets):
        # Reshape according to the specified 3D structure
        # Small boards: (N, 729) -> (N, 1, 9, 9, 9) [Adding channel dim]
        # Assumes flattened data corresponds to Z, Y, X (fastest) within 3x3x3 small cubes,
        # and these cubes are arranged similarly in Z, Y, X (fastest) order.
        self.small_boards = torch.tensor(small_boards, dtype=torch.float32).reshape(-1, 1, 9, 9, 9)

        # Large board: (N, 27) -> (N, 1, 3, 3, 3) [Adding channel dim]
        self.large_board = torch.tensor(large_board, dtype=torch.float32).reshape(-1, 1, 3, 3, 3)

        # Scalars: (N, 2)
        self.scalars = torch.tensor(scalars, dtype=torch.float32)

        # Action Mask: (N, 729)
        self.masks = torch.tensor(masks, dtype=torch.bool) # Use bool for masking efficiency

        # Target: (N,) -> (N,) with Long type for CrossEntropyLoss
        self.targets = torch.tensor(targets, dtype=torch.long)

        print("\nTensor shapes after Dataset creation:")
        print("Small boards tensor:", self.small_boards.shape)
        print("Large board tensor:", self.large_board.shape)
        print("Scalars tensor:", self.scalars.shape)
        print("Masks tensor:", self.masks.shape)
        print("Targets tensor:", self.targets.shape)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.small_boards[idx],
                self.large_board[idx],
                self.scalars[idx]), self.masks[idx], self.targets[idx]

# Instantiate the dataset
dataset = TicTacToe3DDataset(small_boards_data, large_board_data, scalar_data, action_mask_data, target_data)

# Split into training and validation sets
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

print(f"\nDataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True if device != torch.device('cpu') else False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2, pin_memory=True if device != torch.device('cpu') else False)
print(f"Using {os.cpu_count()//2} workers for DataLoaders")

# --- 3. Define the CNN Model ---

class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()

        # Shared Convolutional Layers (as requested)
        # Layer 1: 9x9x9 -> 3x3x3 (for small) | 3x3x3 -> 1x1x1 (for large)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=50, kernel_size=3, stride=3, padding=0)
        self.bn1_s = nn.BatchNorm3d(50) # Separate BN for different stats? Maybe not needed if data dist is similar enough. Let's start shared. Revisit if needed.
        self.bn1_l = nn.BatchNorm3d(50)


        # Layer 2: 3x3x3 -> 3x3x3 (for small) | 1x1x1 -> 1x1x1 (for large)
        self.conv2 = nn.Conv3d(in_channels=50, out_channels=100, kernel_size=1, stride=1, padding=0)
        self.bn2_s = nn.BatchNorm3d(100)
        self.bn2_l = nn.BatchNorm3d(100)

        # Layer 3: 3x3x3 -> 1x1x1 (only for small path)
        self.conv3 = nn.Conv3d(in_channels=100, out_channels=200, kernel_size=3, stride=1, padding=0)
        self.bn3_s = nn.BatchNorm3d(200)

        # Fully Connected Layers
        # Input size: 200 (from small path conv3) + 100 (from large path conv2) + 2 (scalars) = 302
        self.fc1 = nn.Linear(302, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn_fc3 = nn.BatchNorm1d(1024)
        self.fc4_logits = nn.Linear(1024, 729) # Output raw logits

        # Activation
        self.relu = nn.ReLU()

        # Initialize weights (optional but good practice)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_small, x_large, x_scalars):
        # --- Small Board Path ---
        # Layer 1
        out_s1 = self.relu(self.bn1_s(self.conv1(x_small))) # Shape: (N, 50, 3, 3, 3)
        # Layer 2
        out_s2 = self.relu(self.bn2_s(self.conv2(out_s1))) # Shape: (N, 100, 3, 3, 3)
        # Layer 3
        out_s3 = self.relu(self.bn3_s(self.conv3(out_s2))) # Shape: (N, 200, 1, 1, 1)

        # --- Large Board Path ---
        # Layer 1 (Shared conv1)
        out_l1 = self.relu(self.bn1_l(self.conv1(x_large))) # Shape: (N, 50, 1, 1, 1)
        # Layer 2 (Shared conv2)
        out_l2 = self.relu(self.bn2_l(self.conv2(out_l1))) # Shape: (N, 100, 1, 1, 1)

        # --- Flatten and Concatenate ---
        flat_s3 = out_s3.view(out_s3.size(0), -1) # Shape: (N, 200)
        flat_l2 = out_l2.view(out_l2.size(0), -1) # Shape: (N, 100)

        # Concatenate features: small_path, large_path, scalars
        combined = torch.cat((flat_s3, flat_l2, x_scalars), dim=1) # Shape: (N, 302)

        # --- Fully Connected Layers ---
        # Layer 4
        fc1_out = self.relu(self.bn_fc1(self.fc1(combined))) # Shape: (N, 1024)
        # Layer 5
        fc2_out = self.relu(self.bn_fc2(self.fc2(fc1_out))) # Shape: (N, 1024)
        # Layer 6
        fc3_out = self.relu(self.bn_fc3(self.fc3(fc2_out))) # Shape: (N, 1024)
        # Layer 7 (Logits)
        logits = self.fc4_logits(fc3_out) # Shape: (N, 729)

        return logits # Return raw logits, masking and softmax happen outside or in loss

# Instantiate the model and move to device
model = TicTacToeCNN().to(device)
print("\nModel Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")


# --- 4. Loss Function and Optimizer ---

# CrossEntropyLoss expects raw logits. We will apply the mask *before* passing to loss.
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Learning rate scheduler (optional but often helpful)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)


# --- 5. Training Loop ---

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    start_time = time.time()
    for i, (inputs, masks, targets) in enumerate(loader):
        small_boards, large_board, scalars = inputs
        small_boards = small_boards.to(device, non_blocking=True)
        large_board = large_board.to(device, non_blocking=True)
        scalars = scalars.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(small_boards, large_board, scalars) # Shape: (N, 729)

        # Apply mask BEFORE loss calculation
        # Add a large negative number where mask is False (illegal moves)
        masked_logits = logits.masked_fill(~masks, -1e9) # ~masks inverts boolean mask

        # Calculate loss
        loss = criterion(masked_logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * small_boards.size(0)
        _, predicted_actions = torch.max(masked_logits, 1)
        correct_predictions += (predicted_actions == targets).sum().item()
        total_samples += small_boards.size(0)

        # Optional: Print progress within epoch
        # if (i + 1) % 10 == 0:
        #     print(f"  Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_acc, epoch_time

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, masks, targets in loader:
            small_boards, large_board, scalars = inputs
            small_boards = small_boards.to(device, non_blocking=True)
            large_board = large_board.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(small_boards, large_board, scalars)
            masked_logits = logits.masked_fill(~masks, -1e9)
            loss = criterion(masked_logits, targets)

            running_loss += loss.item() * small_boards.size(0)
            _, predicted_actions = torch.max(masked_logits, 1)
            correct_predictions += (predicted_actions == targets).sum().item()
            total_samples += small_boards.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- Training Execution ---
print(f"\nStarting training for {EPOCHS} epochs...")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')

total_train_start_time = time.time()

for epoch in range(EPOCHS):
    train_loss, train_acc, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Update learning rate scheduler
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')
        print(f"  -> New best model saved with Val Loss: {best_val_loss:.4f}")


total_train_end_time = time.time()
print(f"\nTraining finished in {(total_train_end_time - total_train_start_time)/60:.2f} minutes.")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

# --- 6. Plotting Results ---

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
print("\nTraining curves saved to training_curves.png")
plt.show()

# --- Example of loading the best model for inference (optional) ---
# model.load_state_dict(torch.load('best_model_weights.pth'))
# model.eval()
# Now the model is ready for inference using the weights that achieved the best validation loss.