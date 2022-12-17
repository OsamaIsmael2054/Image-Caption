import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from get_loader import get_loader
from model import CNNtoRNN

PATH_TO_MODEL = "./Weights.pth"

def train(num_epochs = 20):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_loader, dataset = get_loader(transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab) + 1 
    num_layers = 1
    learning_rate = 0.01


    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.lookup_indices(["<PAD>"])[0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the ANN
    for name, param in model.encoderCNN.vgg16.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    model.train()
    train_losses = []
    best_loss = 10
    for epoch in range(num_epochs):
        train_loss = 0
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            train_loss += loss
            
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        train_loss /= len(train_loader) 

        if train_loss < best_loss:
            best_loss = train_loss
            print("New Weights Saved...")
            torch.save(model.state_dict(), PATH_TO_MODEL)

        print(f"Epoch {epoch+1}... Loss : {train_loss:0.4f}")
        print("\t " + "________"*7)

if __name__ == "__main__":
    train(num_epochs=10)
