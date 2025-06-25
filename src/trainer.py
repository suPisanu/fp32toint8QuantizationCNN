import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn.quantized as nnq

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 32, (3,3), 1, 1)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((2,2), 2)

        self.conv2 = nn.Conv2d(32, 64, (3,3), 1, 1)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d((2,2), 2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(3136, 128)
        self.dense2 = nn.Linear(128, 10)
        
        self.relu3 = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
    
        x = self.relu3(x)
        x = self.dequant(x)
        return x

if __name__ == "__main__" : 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} device")
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 4096
    learning_rate = 1e-3
    # Create data loaders.
    train_dataloader = DataLoader(training_data,
                                  shuffle=True, 
                                  batch_size=batch_size)
    test_dataloader = DataLoader(test_data, 
                                 shuffle=True, 
                                 batch_size=batch_size)
    # Create NN Model   
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr=learning_rate)
    max_epochs = 10
    n_batch = len(train_dataloader)
    i: int = 0

    with tqdm(total = max_epochs * n_batch) as pbar:
        model.train()
        for epoch in range(max_epochs):
            x: torch.Tensor
            y: torch.Tensor
            y_pred: torch.Tensor
            loss: torch.Tensor
            for batch, (x, y) in enumerate(train_dataloader):
                i = pbar.n + 1
                # move data to gpu
                x, y = x.to(device), y.to(device)
                # compute prediction error
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update logs
                pbar.set_postfix({'i': i, 'epoch': epoch + 1, 'batch': batch + 1, 'loss': loss.item()})
                pbar.update()

    model_fp32 = model.cpu()
    model_fp32.eval()

    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    #model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv1', 'relu1'], ['conv2', 'relu2']])

    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
    #model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

    with torch.no_grad():
        for x, _ in train_dataloader:
            model_fp32_prepared(x)
            break  # just one batch is enough for calibration

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    state_dict = torch.save(model_int8.state_dict(), "model_quantized_state_dict_11.pth")

    scripted_model = torch.jit.script(model_int8)
    scripted_model.save("model_quantized_11.pt")

    ##This Section is for non-quantized model test section##

    #predicted_digits: int = []
    #actual_digits: int = []
    #with torch.no_grad():
    #    x: torch.Tensor
    #    y: torch.Tensor
    #    y_pred: torch.Tensor
    #    loss: torch.Tensor
    #    for batch, (x, y) in enumerate(test_dataloader):
    #        # move data to device
    #        x, y = x.to(device), y.to(device)
    #        # make the predictions and calculate the validation loss
    #        y_pred = model(x)
    #        loss = loss_fn(y_pred, y)
    #        # move data to cpu
    #        predicted_digits += y_pred.argmax(1).detach().cpu().tolist()
    #        actual_digits += y.detach().cpu().tolist()
    #display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(actual_digits, predicted_digits))
    #display.plot()
    #plt.show()

    ##------------------------------------------------------##

    ##This Section is for quantized model test section##

    #Store predictions and ground truth
    #all_preds = []
    #all_labels = []
    #
    ## No gradient needed during evaluation
    #with torch.no_grad():
    #    for x, y in test_dataloader:

    #        #print(f"{x.shape}\n")
    #        #print(f"{y.shape}\n")
    #        
    #        # Quantized model must run on CPU
    #        x = x.cpu()
    #        y = y.cpu()

    #        # Forward pass
    #        output = model(x)
    #        #print(f"{output.shape}\n")
    #        # If model doesn't include LogSoftmax, apply it manually
    #        # Or just use argmax directly if using CrossEntropyLoss
    #        preds = output.argmax(dim=1)
    #        #print(f"{preds.shape}\n")

    #        all_preds.extend(preds.tolist())
    #        all_labels.extend(y.tolist())
    #
    ## Compute confusion matrix
    #cm = confusion_matrix(all_labels, all_preds)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
    #plt.title("Confusion Matrix (Quantized Model)")
    #plt.show()

    ##------------------------------------------------------##