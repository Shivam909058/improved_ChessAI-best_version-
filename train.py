import pandas as pd
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(15, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def create_board_matrix(moves):
    
    board = chess.Board()
    try:
        move_list = moves.split()
        for move in move_list:
            board.push_san(move)
    except:
        return None
    
   
    matrix = np.zeros((8, 8, 12), dtype=np.float32)
    
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_type = str(piece)
            matrix[rank][file][piece_idx[piece_type]] = 1
    
    return add_auxiliary_planes(matrix)

def add_auxiliary_planes(matrix):

    mobility = np.zeros((8, 8, 1), dtype=np.float32)
   
    protection = np.zeros((8, 8, 1), dtype=np.float32)
    
    attacks = np.zeros((8, 8, 1), dtype=np.float32)
    
    return np.concatenate([matrix, mobility, protection, attacks], axis=2)

def preprocess_data(csv_path):
    
    print("Loading chess games dataset...")
    df = pd.read_csv(csv_path)
    
   
    df = df[df['victory_status'].isin(['mate', 'resign', 'outoftime'])]
    print(f"Processing {len(df)} valid games...")
    
    X = []
    y = []
    
    for idx, row in enumerate(df.iterrows()):
        if idx % 1000 == 0:
            print(f"Processing game {idx}/{len(df)}")
        
        board_matrix = create_board_matrix(row[1]['moves'])
        if board_matrix is not None:
            X.append(board_matrix)
            y.append(1 if row[1]['winner'] == 'white' else 0)
    
    
    X = np.array(X).transpose(0, 3, 1, 2)
    y = np.array(y)
    
    return X, y

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'chess_model_best.pth')
            print("Saved best model checkpoint")
        
    return train_losses, val_losses

def main():
    
    X, y = preprocess_data('games.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
    train_dataset = ChessDataset(X_train, y_train)
    test_dataset = ChessDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    model = ChessNet().to(device)
    
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    
    torch.save(model.state_dict(), 'chess_model.pth')
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training complete!")

if __name__ == "__main__":
    main()