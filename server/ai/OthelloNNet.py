import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, random_split
from data_loader import OthelloDataSet
import numpy as np
from tqdm import tqdm
from torch import optim
import time

#device 
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'Device: {DEVICE}')

'''
args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512
'''
#class dotdict cho phép sử dụng chấm để truy nhập các thuộc tính của dictionary  
class dotdict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e


args = dotdict({
    'numIters': 1,            # In training, number of iterations = 1000 and num of episodes = 100
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,      # To control exploration and exploitation
    'updateThreshold': 0.6,   # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200,     # Number of game examples to train the neural networks.
    'numMCTSSims': 15,        # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,       # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'maxDepth': 5,             # Maximum number of rollouts
    'numMCsims': 5,           # Number of monte carlo simulations
    'mc_topk': 3,             # Top k actions for monte carlo rollout

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    # Define neural network arguments
    'lr': 0.001,               # lr: Learning Rate
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'device': DEVICE,
    'num_channels': 512,
})

class ResidualBlock(nn.Module):
    '''The Residual block of ResNet models.'''
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1, stride=strides)

        self.conv2 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1)
        #use 1x1 conv to make input and output have a same shape
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.functional.relu(Y)

class ResNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU(),
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i==0 and not first_block:#nếu ko phải khối Residual Block đầu tiên x2 channels và giảm 2 lần height, width
                blk.append(ResidualBlock(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResidualBlock(num_channels))
        return nn.Sequential(*blk)

    def __init__(self, arch, lr=0.1, num_classes=10):
        super(ResNet, self).__init__()
        
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))
    
    def forward(self, x):
        for block in self.net:
            x = block(x)
        return x

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

# 0) prepare data  
othello_datasets = OthelloDataSet(csv_file='othello_tensor_datasets.csv')
Train_size = int(0.8*len(othello_datasets))
val_size = int(0.1*len(othello_datasets))
test_size = len(othello_datasets) - Train_size - val_size

Train_datasets, val_datasets, Test_datasets = random_split(othello_datasets, [Train_size, val_size, test_size])

Train_loader = DataLoader(Train_datasets, batch_size=args.batch_size, shuffle=True)
Val_loader = DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
Test_loader = DataLoader(Test_datasets, batch_size=args.batch_size, shuffle=False)

'''
TODO: dùng minimax để tạo ra các nhánh --> sau đó từ mỗi nhánh mình cho chạy qua CNN để tính xs chiến thắng
'''

# 1) model

model = OthelloNNet().to(DEVICE)

# 2) loss and optimizer
# 3) training loop


#class ValueNetwork(NeuralNet):

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

  def train(self, games):
    """
    Args:
      games: list
        List of examples with each example is of form (board, pi, v)
    """
    optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
    for examples in games:
      for epoch in range(args.epochs):
        print('EPOCH ::: ' + str(epoch + 1))
        self.nnet.train()
        v_losses = []   # To store the losses per epoch
        batch_count = int(len(examples) / args.batch_size)  # len(examples)=200, batch-size=64, batch_count=3
        t = tqdm(range(batch_count), desc='Training Value Network')
        for _ in t:
          sample_ids = np.random.randint(len(examples), size=args.batch_size)  # Read the ground truth information from MCTS simulation using the loaded examples
          boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))  # Length of boards, pis, vis = 64
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

          # Predict
          # To run on GPU if available
          # hàm contiguous() tạo ra một bản sao liên tục trong bộ nhớ khi một tensor có thể không liên tục
          # hoạt động hiệu quả khi làm việc với GPU
          boards, target_vs = boards.contiguous().to(args.device), target_vs.contiguous().to(args.device)

          # Compute output
          _, out_v = self.nnet(boards)
          l_v = self.loss_v(target_vs, out_v)  # Total loss(MSE loss)

          # Record loss
          v_losses.append(l_v.item())
          t.set_postfix(Loss_v=l_v.item())

          # Compute gradient and do SGD step
          optimizer.zero_grad()
          l_v.backward()
          optimizer.step()

  def predict(self, board):
    """
    Args:
      board: np.ndarray
        Board of size n x n [6x6 in this case]

    Returns:
      v: OthelloNet instance
        Data of the OthelloNet class instance above;
    """
    # Timing
    start = time.time()

    # Preparing input
    board = torch.FloatTensor(board.astype(np.float64))
    board = board.contiguous().to(args.device)
    board = board.view(1, self.board_x, self.board_y)
    self.nnet.eval()
    with torch.no_grad():
        _, v = self.nnet(board)
    return v.data.cpu().numpy()[0]

  def loss_v(self, targets, outputs):
    """
    Args:
      targets: np.ndarray
        Ground Truth variables corresponding to input
      outputs: np.ndarray
        Predictions of Network

    Returns:
      MSE Loss averaged across the whole dataset
    """
    # Mean squared error (MSE)
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    save_model_checkpoint(folder, filename, self.nnet)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    load_model_checkpoint(folder, filename, self.nnet, args.device)
    
    
    
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
        for epoch in tqdm(range(args.num_epochs), desc='Training Value Network'):
            #training
            self.nnet.train()
            train_correct =0
            train_total =0
            train_loss_epoch = 0.0
            for i, (images, labels) in enumerate(Train_loader):
                images = images.contiguous().to(args.device)
                labels = labels.contiguous().to(args.device)
            
                #forward pass
                _, outputs = self.nnet(images)
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
                

