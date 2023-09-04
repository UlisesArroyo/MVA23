
#!/bin/bash


python main.py --pathDataset="./datasets/properlyFinal" --batchSize=16 --lr=0.001 --modelo="convnext"

python main.py --pathDataset="./datasets/properlyFinal" --batchSize=16 --lr=0.001 --modelo="resnet50"

python main.py --pathDataset="./datasets/properlyFinal" --batchSize=16 --lr=0.001 --modelo="resnext50"

python main.py --pathDataset="./datasets/properlyFinal" --batchSize=16 --lr=0.001 --modelo="mobilenetv3"

python main.py --pathDataset="./datasets/properlyFinal" --batchSize=16 --lr=0.001 --modelo="efficientnet"

python main.py --pathDataset="./datasets/properlyFinal" --batchSize=16 --lr=0.001 --modelo="internimage"


python main.py --pathDataset="./datasets/TFM-07-23" --batchSize=16 --lr=0.001 --modelo="convnext"

python main.py --pathDataset="./datasets/TFM-07-23" --batchSize=16 --lr=0.001 --modelo="resnet50"

python main.py --pathDataset="./datasets/TFM-07-23" --batchSize=16 --lr=0.001 --modelo="resnext50"

python main.py --pathDataset="./datasets/TFM-07-23" --batchSize=16 --lr=0.001 --modelo="mobilenetv3"

python main.py --pathDataset="./datasets/TFM-07-23" --batchSize=16 --lr=0.001 --modelo="efficientnet"

python main.py --pathDataset="./datasets/TFM-07-23" --batchSize=16 --lr=0.001 --modelo="internimage"