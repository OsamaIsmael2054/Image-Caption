# Basic imports
import os  
import pandas as pd  
import torch
from torch.utils.data import DataLoader, Dataset
import pickle

#handle image
from PIL import Image  # Load img
import torchvision.transforms as transforms

#handle Text
from torch.nn.utils.rnn import pad_sequence  
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

PATH_TO_DATA = "D:\\Datasets\\flickr8k"


class Vocabulary:
    def __init__(self, captions, freq_threshold=5):
        self.tokenizer = get_tokenizer("basic_english")
        self.freq_threshold = freq_threshold
        self.vocab = self.build_vocabulary(captions)

    def yield_tokens(self, captions: list):
        for i in range(len(captions)):
            yield self.tokenizer(captions[i])

    def build_vocabulary(self,captions: list):
        vocab = build_vocab_from_iterator(self.yield_tokens(captions),
                        specials=["<PAD>","<UNK>","<SOS>","<EOS>"], min_freq = self.freq_threshold)

        vocab.set_default_index(vocab["<UNK>"])
        return vocab

    def numericalize(self, text):
        return self.vocab(self.tokenizer(text))



class FlickrDataset(Dataset):
    def __init__(self, transform=None, freq_threshold=5):

        data = pd.read_csv(os.path.join(PATH_TO_DATA,"captions.txt"))

        self.root_dir = PATH_TO_DATA + "\\images"
        self.transform = transform

        # Get img, caption columns
        self.imgs = data["image"]
        self.captions = data["caption"]

        #build vocab
        self.vocabulary = Vocabulary(self.captions.tolist(), freq_threshold)
        self.vocab = self.vocabulary.vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = self.vocab.lookup_indices(["<SOS>"])
        numericalized_caption += self.vocabulary.numericalize(caption)
        numericalized_caption.append(self.vocab.lookup_indices(["<EOS>"])[0])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(transform, batch_size=32, shuffle=True, pin_memory=True):

    dataset = FlickrDataset(transform=transform)

    pad_idx = dataset.vocab.lookup_indices(["<PAD>"])[0]

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx))

    return loader, dataset