import torch
import torch.nn as nn
import torch.nn.functional as F

#feedforward nn, takes input vector (784 cause 28*28), pushes it to hidden layers and outputs 10 scores
class MLP(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        #Linear layer, connects every input to every output
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        #randomly turns off some activation during training
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        #x starts as image tensor then flattens into (N,784)
        x = x.view(x.size(0), -1)
        #layer1
        x = F.relu(self.fc1(x)) #relu turns negative z vals to 0 to turn off some some neurons and change with paths are active
        x = self.drop(x)        #z is = Wx + b (weight*input + bias)
        #layer2
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        #out
        return self.fc3(x)

def load_model(model_path="mnist_mlp_best.pt", dropout=0.3, device=None):
    #decide where model will run(gpu or cpu)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    #creates new model object with layers and random weights 
    model = MLP(dropout=dropout).to(device)
    #load trained weights and biases from the .pt file to the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device