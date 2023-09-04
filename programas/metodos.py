from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
from tqdm import tqdm
import os 



def train_model(model, name_model, name_dataset , lr, criterion, optimizer, scheduler,  dataloders, use_gpu,dataset_sizes, num_epochs=30):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            #print(dataloders[phase])
            #exit()
            # Iterate over data.
            process_bar = tqdm(enumerate(dataloders[phase]), total=len(dataloders[phase]))
            for _,data in process_bar:
                #print("data: ", data)
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #print("outputs: ", outputs)
                loss = criterion(outputs, labels)
                #print("loss: ", loss)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            process_bar.set_description_str('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))

            ruta = './resultados/entrenamiento/'+ name_dataset + '/' + name_model+'/lr-' + str(lr) +'/'
            with open(ruta+'res.txt', 'w'):
                pass
            if not os.path.exists(ruta):
                os.makedirs(ruta)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                with open(ruta+'res.txt', 'w') as archivo:
                    archivo.write("Best acc: {:4f}".format(best_acc) + " en la epoca: " +  str(epoch+1))
                torch.save(model, ruta +'best_model.pth')

            
            if epoch == num_epochs - 1:
                torch.save(model,ruta + 'last_model.pth')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




def testing(model):
    seleccion = 'test'
    dir_out = 'testeos/TFM-'+seleccion+'-fineTuning-maskedFaceNet-4-0.001-rev-porcentajes'
    model.eval()
    
    trues = []
    preds = []
    #print("dataloader:", dataloders[seleccion])

    running_corrects = 0.0
    #print("DATALOADERS: ", dataloders[seleccion])

    for i, (images, labels) in enumerate(dataloders[seleccion],0):
        #labels = labels.squeeze()
        print("labels:", i,labels)
        #print("images [shape]:", images.shape)
        #print("images [dtype]:", images.dtype)
        #print("Imagenes: ", images)

        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        #print("outputs",outputs.tolist())
        #print("outputs",outputs.shape)
        
        # preds ) torch.argmax(outputs, dim=1)
        print("preds", torch.argmax(outputs, dim=1))
        
        #torch.argmax(pred, dim=1)[0] 
        pred = torch.argmax(outputs, dim=1)
        #print("Pred --: ", pred)
        
        porcentajes = [outputs[x][y].tolist() for x, y in enumerate(pred)]

        print("porcentajes",porcentajes)
        print("pred:",pred.tolist())
        #print("pred:",pred.shape)


        running_corrects += torch.sum(pred == labels.data).to(torch.float32)
        if len(trues) == 0:
            trues = labels
            preds = pred
        else:
            trues=torch.cat([trues, labels],dim=0)
            preds=torch.cat([preds, pred],dim=0)
        
        
        #exit()
        
        for x in range(len(labels)):
            #print(i*8+x)
            certeza = eleccion(porcentajes[x])
            print()
            sample_fname ,_ = dataloders[seleccion].dataset.samples[i*8+x]
            #print("sample_fname",sample_fname)
            #print("--- preds: ", str(pred[x].tolist()))
            #shutil.copy(sample_fname, dir_out + "/"+str(pred[x].tolist()))

            shutil.copy(sample_fname, dir_out + "/"+str(pred[x].tolist()) + '/' +str(certeza))
            print(dir_out + "/"+str(pred[x].tolist()) + '/' +str(certeza))
        #if i ==3:
        #    exit()
    epoch_acc = running_corrects / dataset_sizes[seleccion]

    print('Acc: {:.4f}'.format(epoch_acc))


    trues = trues.tolist()
    preds = preds.tolist()
    #plot(trues,preds)
    #print("trues:",type(trues))
    #print("preds:",type(preds)) 
    #print("trues [shape]:",len(trues))
    #print("preds [shape]:",len(preds))
    #preds.append(int(torch.argmax(pred, dim=1)[0]))
        
        #preds.append(int(pred))
        #trues.append(int(labels))
    
    
    #print("preds",preds)
    
