from PIL import Image
import torch
from torchvision import transforms

from frcnn import FasterRCNN


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint = torch.load('model/alfred_model_e05.pth')
    model = FasterRCNN(num_classes=2, training=False, checkpoint=checkpoint, save_features=False)
    model.to(device)

    model.eval()

    image = Image.open('data/alfred-i/image/target/000001.jpg')
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image)
    image = [image.to(device)]

    output = model(image)


if __name__=='__main__':
    main()
