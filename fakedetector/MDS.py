from .audiornn import AudioRNN
from . import FakeDetector
from .audiornn.augmentation import Scale, ToTensor, Normalize
import torch
from torchvision import transforms
import os, sys
from PIL import Image


class MDSFakeDetector(FakeDetector):

    def __init__(self, resnet='resnet18', finaldim=1024, dropout=0.5,**kwargs):
        super().__init__(**kwargs)
        self.transform = transforms.Compose([
        Scale(size=(self.imgdim, self.imgdim)),
        ToTensor(),
        Normalize()
        ])

        self.resnet = resnet
        self.finaldim = finaldim
        self.dropout = dropout
        self.model = None
        self.fakeclass = 0


    def initializemodel(self):
        self.model = AudioRNN(img_dim=self.imgdim, network=self.resnet, num_layers_in_fc_layers=self.finaldim, dropout=self.dropout, winLength=self.winlength)
        self.model = torch.nn.DataParallel(self.model)

    def load_model(self,path):
        self.initializemodel()
        if os.path.isfile(path):
            print("=> loading testing checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                sys.exit()
            print("=> loaded testing checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        else:
            print("Saved model path not found")
            sys.exit()
        self.model = self.model.to(self.device)

    def preprocess(self, chunkslist):

        batchlist = []
        for chunk in chunkslist:
            chunk = [Image.fromarray(image) for image in chunk]
            t_seq = self.transform(chunk)  # apply same transform

            (C, H, W) = t_seq[0].size()
            t_seq = torch.stack(t_seq, 0)
            t_seq = t_seq.view(1, self.winlength, C, H, W).transpose(1, 2)
            batchlist.append(t_seq)

        return torch.stack(batchlist)

    def predict(self,chunkslist):
        """

        :param chunkslist: list of chunks
        :return: list of fake scores [B,2]
        """
        videoseqs=self.preprocess(chunkslist)
        self.model.eval()
        with torch.no_grad():
            seqbatch = videoseqs.to(self.device)
            interm = self.model.module.forward_lip(seqbatch)
            logits = self.model.module.final_classification_lip(interm)
            prob = torch.nn.functional.softmax(logits, 1)
            fakescores = prob.cpu().numpy()[:,self.fakeclass]#dim [B,2]
        return fakescores