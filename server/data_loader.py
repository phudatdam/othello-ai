import numpy as np 
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
import ast

path = 'C:/Users/THE ANH/Desktop/code/prj_game_ai/othello_tensor_datasets.csv'

class OthelloDataSet(Dataset):
    def __init__(self, csv_file, transform = None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        game_id = self.data.iloc[index, 0]
        winner = self.data.iloc[index, 1]
        winner = np.array(winner)
        board_state_str = self.data.iloc[index, 2]

        # Chuyển đổi chuỗi biểu diễn tensor thành tensor PyTorch
        # Sử dụng ast.literal_eval để đánh giá một cách an toàn chuỗi Python chứa literal
        #board_state_list = ast.literal_eval(board_state_str)
        #board_state = torch.tensor(np.array(board_state_list, dtype=np.float32)) # Đảm bảo kiểu dữ liệu phù hợp

        sample = {'game_id': game_id, 'winner': torch.tensor(winner, dtype=torch.float32).unsqueeze(0), 'board_state': board_state_str}

        if self.transform:
            sample['board_state'] = self.transform(sample['board_state'])

        return sample
    
    
if __name__ == '__main__':
    othello_dataset = OthelloDataSet(csv_file='othello_tensor_datasets.csv')
    
    sample = othello_dataset[50]
    print("Game ID:", sample['game_id'])
    print("Winner:", sample['winner'])
    print("Board State:\n", sample['board_state'])
    #print("Kích thước Board State:", sample['board_state'].shape)
    print(othello_dataset.__len__())

    # Tạo DataLoader
    dataloader = DataLoader(othello_dataset, batch_size=2, shuffle=True)
    
    for i, bat in enumerate(dataloader):
        if i==2:
            print(bat.shape)
    '''
    # Lặp qua DataLoader
    for batch in dataloader:
        print("\nBatch:")
        print("Game IDs:", batch['game_id'])
        print("Winners:", batch['winner'])
        print("Board States:\n", batch['board_state'])
        #print("Kích thước Board States:", batch['board_state'].shape)
        break
    '''

