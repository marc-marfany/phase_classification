import torch.nn as nn

# Here I recreate the architecture of the paper Phase Detection with Neural Networks: Interpreting the Black Box (https://arxiv.org/pdf/2004.04711).

class CNN_1D_FH_half_filling(nn.Module):
    """In this class I will implement the main architecture of the paper.
    """
    # For 1D convolution layers and Average Pooling 1D
    # w_out=(w_input - kernel_size+ 2*padding)/stride +1
    
    # For 1D MaxPool Layer 
    # w_out= (w_input - kernel_size +2*padding)/ stride +1
    def __init__(self):
        super(CNN_1D_FH_half_filling,self).__init__()
        
        self.layer_1=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=1, stride=1), # (924 - 15 +2*1)/1 +1 = 912
            nn.MaxPool1d(kernel_size=4, stride=4), # (912 - 4 - 0)/4  + 1 = 228
            nn.ReLU(),
        )
        
        self.layer_2=nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=8, kernel_size=5, padding=0, stride=1), # (228 - 5 + 0)/1 +1 = 224
            nn.MaxPool1d(kernel_size=4, stride=4), # (224 -4 - 0)/4 + 1 = 56
            nn.ReLU(),
        )

        self.layer_3=nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=10, kernel_size=3, padding=1, stride=1), # (56 - 3 + 2*1)/1 +1 = 56
            nn.AvgPool1d(kernel_size=4, stride=4), # (56 - 4 - 0)/4  +1 = 14
            nn.ReLU(),
        )

        self.layer_4=nn.Linear(in_features= 10 * 14, out_features= 2) # Output 2 possible classes
        self.dropout=nn.Dropout()
    
    def forward(self, x):
        """ Forward pass """
        # The shape of the data is x.shape=(batch_size, 1 , 924)
        x=self.layer_1(x)
        
        # The shape of the data is x.shape=(batch_size, 5, 228)
        x=self.layer_2(x)
        
        # The shape of the data is x.shape=(batch_size, 8, 56)
        x=self.layer_3(x)

        # The shape of the data is x.shape=(batch_size, 10,14)
        # Reshape x to flatten the first and second dimension
        x=x.flatten(start_dim=1)

        # Activate dropout
        x=self.dropout(x)

        # last linear layer
        x=self.layer_4(x)

        return x