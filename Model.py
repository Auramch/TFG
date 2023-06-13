
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import os
import jsonlines
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import time
from sentence_transformers import SentenceTransformer
from torchvision.models import ResNet18_Weights
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from transformers import AutoTokenizer,AutoModel
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]
        # x = [batch size, height * width]

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  #mirar que arriba y si se esta pasant be
        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2

class MyDataSet(Dataset):
    
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        if torch.is_tensor(id):
            id = id.tolist()
        if (id != 0):
            img_name = os.path.join(self.root_dir,
                              str(list(self.data.keys())[id])+'.jpg')
        
            with Image.open(img_name) as image:
                hypotesis = {}
                for sentence in self.data[list(self.data.keys())[id]]:
                    label = self.data[list(self.data.keys())[id]][sentence]
                    #sentence = sentence.astype('float').reshape(-1, 2)
                    if (label == "entailment" ):
                        logit = [1.0,0.0,0.0]
                    elif(label == "contradiction"):
                        logit = [0.0,1.0,0.0]
                    elif (label == "neutral"):
                        logit = [0.0,0.0,1.0]
                
                    if self.transform:
                        dataset = self.transform((image,  sentence,  logit))
                    return dataset


class Tensor(object):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, sample):
        image, hypothesis, label = sample[0], sample[1], sample[2]
        #image = np.transpose(image,(2, 0, 1))
        
        transform= transforms.Compose([transforms.Resize((224, 224)),transforms.PILToTensor()])
        image = transform(image)
        #convert_tensor = transforms.ToTensor()
        #image = convert_tensor(image)
        #hypothesis = self.le.fit_transform(hypothesis)
        hypothesis = tokenizer(hypothesis,padding='max_length', max_length=50, truncation=True, return_tensors='pt')
        #hypothesis = np.array(hypothesis['input_ids'])
        return image, hypothesis['input_ids'][0], hypothesis['token_type_ids'][0], hypothesis['attention_mask'][0], torch.from_numpy(np.array(label))


def dataset():
    filename = '/data-fast/107-data4/amartinez/SNLI-VE/snli_1.0/snli_ve/snli_ve_train.jsonl'
    data = {}
    with jsonlines.open(filename) as jsonl_file:
        for line in jsonl_file:
            imageID = str(line['Flickr30K_ID'])
            hypothesis = str(line['sentence2'])
            label = str(line['gold_label'])
    
            if imageID not in data.keys():
                data[imageID] = {hypothesis: label}
            else:
                data[imageID][hypothesis] = label
        return data

if __name__ == '__main__':

    le = preprocessing.LabelEncoder()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    transformed_dataset = MyDataSet(data = dataset(), root_dir='/data-fast/107-data4/amartinez/SNLI-VE/flickr30k_images', 
                           transform = Tensor(tokenizer))
    #transforms.Resize((224, 224)),  
   # transforms.RandomResizedCrop(224),
   # transforms.RandomHorizontalFlip(),
    
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dataloader_dataset = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0) #augmentar el batch a l'hora de entrenar

    
    model_image = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  
    model_image.fc = nn.Identity()
    model_sentence = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model_sentence_reduct =  nn.Linear(768, 512)
    model_f = MLP(1024, 3)
    optimizer = optim.SGD(model_f.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10 
    start_time = time.time() 
    for epoch in range(num_epochs): 
        print("Epoch {} running".format(epoch)) 
        model_f.train()   
        running_loss = 0.  
        running_corrects = 0 
 
        for i, (inputs_image, input_ids, token_type_ids,attention_mask, label) in enumerate(dataloader_dataset):
            
            outputs_image = model_image(inputs_image.float())

            inputs_sentence = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
            outputs_sentence = model_sentence(**inputs_sentence)
            outputs_sentence = model_sentence_reduct(outputs_sentence.pooler_output)

            outputs_concat = torch.cat([outputs_image, outputs_sentence], axis=-1)
            output,out_feats = model_f(outputs_concat)

            #_, preds = torch.max(output, 1)
            loss = tf.nn.softmax_cross_entropy_with_logits(output, label)
            #loss_value = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item() * outputs_concat.size(0)
            running_corrects += torch.sum(preds == label.data)
        epoch_loss = running_loss / len(outputs_concat)
        epoch_acc = running_corrects / len(outputs_concat) * 100.
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() -start_time))
        
        """ Testing Phase """
        model_f.eval()

        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for i, (inputs_image, input_ids, token_type_ids,attention_mask, label) in enumerate(dataloader_dataset):
            
                outputs_image = model_image(inputs_image.float())
                inputs_sentence = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
                outputs_sentence = model_sentence(inputs_sentence)
                outputs_sentence = model_sentence_reduct(outputs_sentence)
                outputs_concat = nn.concatenate([outputs_image, outputs_sentence], axis=-1)
                output = model_f(outputs_concat)
                _, preds = torch.max(output, 1)
                loss = tf.nn.softmax_cross_entropy_with_logits(output, label)
        
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item() * outputs_concat.size(0)
                running_corrects += torch.sum(preds == label.data)
        epoch_loss = running_loss / len(outputs_concat)
        epoch_acc = running_corrects / len(outputs_concat) * 100.
        epoch_loss = running_loss / len(outputs_concat)
        epoch_acc = running_corrects / len(outputs_concat) * 100.
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))

    torch.save(model_f.state_dict(), 'model_10epoch_4batch.pth')
