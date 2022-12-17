from model import CNNtoRNN
import torch
import pickle
import torchvision.transforms as transforms
from PIL import Image 

def prepare_image(path_to_image):

    img = Image.open(path_to_image).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    img = transform(img).unsqueeze(0)
    return img

def caption_image(model, itos, image, max_length=50):
    result_caption = []

    with torch.no_grad():
        x = model.encoderCNN(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            x = model.decoderRNN.embed(predicted).unsqueeze(0)

            if itos[predicted.item()] == "<EOS>":
                break

    return [itos[idx] for idx in result_caption]

def load_model(itos):
    embed_size = 256
    hidden_size = 256
    vocab_size = len(itos) + 1 
    num_layers=1

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers=1)

    model.load_state_dict(torch.load("Weights.pth"))
    model.eval()
    return model


def predict(PATH_TO_IMG):
    with open("itos.pkl","rb") as f:
        itos = pickle.load(f)

    img = prepare_image(PATH_TO_IMG)
    model = load_model(itos)
    result = caption_image(model,itos,img)
    return result

if __name__ == '__main__':
    path_to_img = input("Please enter path to the image: ")
    print(f"OUTPUT is :{predict(path_to_img)[1:-1]}")
