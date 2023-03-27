import torch
import torchvision.transforms as transforms

from PIL import Image
from torch import float16, float32
from torch.utils.data import DataLoader
from torchvision.io import read_image, decode_image, decode_jpeg, encode_jpeg

from garbageNN import CNN
import garbageNN

if __name__ == "__main__":
    img = read_image("test_image/2.jpg")/255
    loader = DataLoader(dataset=img, batch_size=1, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = garbageNN.CNN(in_channels=3, num_classes=6)
    model = garbageNN.load_model(model)
    model.to(device)
    class_name = ('картон', 'стекло', 'металл', 'бумага', 'пластик', 'прочий мусор')
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device=device)
            print(x )
            scores = model(x)
            print(scores)
            _, predictions = scores.max(1)
