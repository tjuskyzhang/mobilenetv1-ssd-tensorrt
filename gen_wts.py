import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

def export_as_weights(net, path_to_wts="models/mobilenet-v1-ssd.wts"):
    """ save the model weights """
    f = open(path_to_wts, 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    print("Saved weights at ", path_to_wts)

def main():
    print('cuda device count: ', torch.cuda.device_count())
    DEVICE = 'cuda:0'
    class_names = [name.strip() for name in open('models/voc-model-labels.txt').readlines()]

    image = torch.ones(1, 3, 300, 300).to(DEVICE)

    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    net.load('models/mobilenet-v1-ssd-mp-0_675.pth')
    net = net.to(DEVICE)

    net = net.eval()
    scores, boxes = net(image)
    print(net(image))
    print("Input shape ", image.shape)
    print("Scores shape ", scores.shape)
    print("Boxes shape ", boxes.shape)

    export_as_weights(net)

if __name__ == '__main__':
    main()
