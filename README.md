# Image Caption

Generate Caption for an image

## Model
Encoder : a pretrained vgg16 that is used to extract features from a given image
Decoder : lstm layer followed by linear layer that is used to generate text

## Files

```python
train.py: used to train the model
prediction.py: generate prediction by give it the path to an image
get_loader.py: used to prepare dataset and dataloader
stoi.pkl, itos.pkl: the vocabulary
```

## Dataset

Download the dataset used: https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb Then set images folder, captions.txt inside a folder Flickr8k.

