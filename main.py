from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import programas.modelos as modelos
import programas.metodos as metodos

def eleccion(name_model, numClases):
    if name_model == "convnext":
        return modelos.convNext(args.numClases)
    elif name_model == "resnet50":
        return modelos.Resnet50(numClases)
    elif name_model == "resnext50":
        return modelos.ResNeXt50(numClases)
    elif name_model == "mobilenetv3":
        return modelos.MobileNetV3(numClases)
    elif name_model == "efficientnet":
        return modelos.EfficientNet(numClases)
    elif name_model == "internimage":
        return modelos.interImage(numClases)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Hiperparametros")
    parser.add_argument("--pathDataset", type=str, default="C:/Users/Arroy/Desktop/chambaDocs/MVA23/datasets/properlyFinal", help="Dirección del dataset")
    parser.add_argument("--batchSize", type=int, default=8, help="Tamaño del lote")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate del modelo")
    parser.add_argument("--epochs", type=int, default=30, help="Epocas que entrena el modelo")
    parser.add_argument("--train", type=str, default="entrenar", help="Status de lo que se realizara con el modelo [entrenar, testear]")
    parser.add_argument("--modelo", type=str, default="convnext", help="El modelo que será entre")
    parser.add_argument("--cargarModel", type=bool, default=False , help="Estado para cargar o no el modelo")
    parser.add_argument("--pathModel", type=str, default="/home/server-1/Desktop/MVA23/modelos/primerEntrenamiento/convNext+MaskedFaceNet+30epochs+lr0.00001/transferLearning/best_model.pth", help="Ruta del modelo a cargar")
    parser.add_argument("--numClases", type=int, default=3 , help="Cantidad de salidas que tendra ")
    args = parser.parse_args()

    #Transformaciones para MaskedFace-Net
    data_transforms = {
        'train': transforms.Compose([
            #Transformaciones para Properly
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    name_dataset = os.path.basename(args.pathDataset)
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.pathDataset, x), data_transforms[x]) for x in ['train', 'val']}
    
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batchSize,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    print(dataset_sizes)


    use_gpu = torch.cuda.is_available()

    if args.train == 'entrenar':
        model_ft, name_model = eleccion(args.modelo, args.numClases)
        #model_ft, name_model = modelos.convNext(args.numClases)
    elif args.train == 'testear':
        model_ft = torch.load(args.pathModel)

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = metodos.train_model(model=model_ft,
                            name_model=name_model,
                            name_dataset = name_dataset,
                            lr=args.lr,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           dataloders=dataloders,
                           use_gpu=use_gpu,
                           dataset_sizes=dataset_sizes,
                           num_epochs=args.epochs)

    

"""
python main.py --pathDataset="C:/Users/Arroy/Desktop/chambaDocs/MVA23/datasets/properlyFinal" --train "entrenar" --cargarModel False
"""