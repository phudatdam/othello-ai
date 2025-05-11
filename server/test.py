from torch.utils.data import DataLoader, random_split
from data_loader import OthelloDataSet
from ai.OthelloNNet import args
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

othello_datasets = OthelloDataSet(csv_file='othello_tensor_datasets.csv')
Train_size = int(0.8*len(othello_datasets))
val_size = int(0.1*len(othello_datasets))
test_size = len(othello_datasets) - Train_size - val_size
Train_datasets, val_datasets, Test_datasets = random_split(othello_datasets, [Train_size, val_size, test_size])

Train_loader = DataLoader(Train_datasets, batch_size=args.batch_size, shuffle=True)
Val_loader = DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
Test_loader = DataLoader(Test_datasets, batch_size=args.batch_size, shuffle=False)

class OthelloNNet(nn.Module):

  def __init__(self, game, args):
    """
    Initialise game parameters

    Args:
      game: OthelloGame instance
        Instance of the OthelloGame class above;
      args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512
    """
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.args = args

    super(OthelloNNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=args.num_channels,
                           kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=args.num_channels,
                           out_channels=args.num_channels, kernel_size=3,
                           stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=args.num_channels,
                           out_channels=args.num_channels, kernel_size=3,
                           stride=1)
    self.conv4 = nn.Conv2d(in_channels=args.num_channels,
                           out_channels=args.num_channels, kernel_size=3,
                           stride=1)

    self.bn1 = nn.BatchNorm2d(num_features=args.num_channels)
    self.bn2 = nn.BatchNorm2d(num_features=args.num_channels)
    self.bn3 = nn.BatchNorm2d(num_features=args.num_channels)
    self.bn4 = nn.BatchNorm2d(num_features=args.num_channels)

    self.fc1 = nn.Linear(in_features=args.num_channels * (self.board_x - 4) * (self.board_y - 4),
                         out_features=1024)
    self.fc_bn1 = nn.BatchNorm1d(num_features=1024)

    self.fc2 = nn.Linear(in_features=1024, out_features=512)
    self.fc_bn2 = nn.BatchNorm1d(num_features=512)

    self.fc3 = nn.Linear(in_features=512, out_features=self.action_size)

    self.fc4 = nn.Linear(in_features=512, out_features=1)

  def forward(self, s):
    """
    Controls forward pass of OthelloNNet

    Args:
      s: np.ndarray
        Array of size (batch_size x board_x x board_y)

    Returns:
      prob, v: tuple of torch.Tensor
        Probability distribution over actions at the current state and the value
        of the current state.
    """
    s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
    s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
    s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
    s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
    s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
    s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

    s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
    s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

    pi = self.fc3(s)  # batch_size x action_size
    v = self.fc4(s)   # batch_size x 1
    # Returns probability distribution over actions at the current state and the value of the current state.
    return F.log_softmax(pi, dim=1), torch.tanh(v)

class ValueNetwork(nn):

    def __init__(self, game):
        """
        Args:
            game: OthelloGame
                Instance of the OthelloGame class above
        """
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.nnet.to(args.device)

    def train(self, data):
        """
        Args:
            games: list
                List of examples with each example is of form (board, pi, v)
        """
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        # khởi tạo các list lưu trữ 
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        
        #save_folder
        save_folder = 'C:/Users/THE ANH/Desktop/code/prj_game_ai/othello-ai/server/hist'
        
        # 3) training loop
        n_total_train = len(Train_loader)
        for epoch in tqdm(range(args.num_epochs)):
            #training
            self.nnet.train()
            train_correct =0
            train_total =0
            train_loss_epoch = 0.0
            for i, (images, labels) in enumerate(Train_loader):
                images = images.to(args.device)
                labels = labels.to(args.device)
            
                #forward pass
                outputs = self.nnet(images)
                loss = criterion(outputs, labels)

                #backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #accuracy ở batch
                train_loss_epoch += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            #accuray qua mỗi epoch    
            train_accuracy = 100 * train_correct / train_total  
            train_loss_epoch /= train_total
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss_epoch)
            
            #test on validation test
            self.nnet.eval()
            val_correct = 0
            val_total=0
            val_loss_epoch = 0.0
            with torch.no_grad():
                for images_val, labels_val in Val_loader:
                    images_val = images_val.to(args.device)
                    labels_val = labels_val.to(args.device)
                    outputs_val = self.nnet(images_val)
                    loss_val = criterion(outputs_val, labels_val)

                    val_loss_epoch += loss_val.item() * images_val.size(0)
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    val_total += labels_val.size(0)
                    val_correct += (predicted_val == labels_val).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            val_loss_epoch /= val_total
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss_epoch)

            file_name = f'ResNet_cifar10_epoch{epoch+1}.pth'
            full_path = os.path.join(save_folder, file_name)
            torch.save(self.nnet.state_dict(), full_path) # Provide the model's state_dict and a file path
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss_epoch:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy:.2f}%')
                




