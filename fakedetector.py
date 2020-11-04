import threading
from queue import Queue
from PIL import Image
from PyQt5.QtCore import pyqtSignal, QObject
import torch
from torchvision import transforms
from augmentation import Scale, ToTensor, Normalize
from models import AudioRNN
import os, sys

'''
Deep learning models
'''
class MDSFakeDetector(object):

    def __init__(self, imgdim= 224, resnet='resnet18', device='cpu', finaldim=1024, dropout=0.5, winLength=30, overlap=0):
        self.imgdim = imgdim
        self.winlength = winLength
        self.overlap = overlap
        self.transform = transforms.Compose([
        Scale(size=(self.imgdim, self.imgdim)),
        ToTensor(),
        Normalize()
        ])
        self.device = device
        self.resnet = resnet
        self.finaldim = finaldim
        self.dropout = dropout
        self.model = None
        self.fakeclass = 0
        self.initializemodel()

    def initializemodel(self):
        self.model = AudioRNN(img_dim=self.imgdim, network=self.resnet, num_layers_in_fc_layers=self.finaldim, dropout=self.dropout, winLength=self.winlength)
        self.model = torch.nn.DataParallel(self.model)

    def loadcheckpoint(self,path):
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
        :return: list of fake scores
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


'''
Classes to handle detection
'''


class FakeDetectorSignals(QObject):
    """
    Defines the signals available from a running thread.

    Supported signals are:

    scoreUpdate
        `tuple` (timestamp,score)
    dataProcessedUpdate
        `tuple` (processed count,captured count)
    """
    scoreUpdate = pyqtSignal(tuple)
    dataProcessedUpdate = pyqtSignal(tuple)
    statusUpdate = pyqtSignal(str)


class FakeDetectionThreading(object):
    def __init__(self, faceDetector, fakeDetector, screenCapture, missLimit=20):
        self.faceDetector = faceDetector
        self.fakeDetector = fakeDetector
        self.sc = screenCapture
        self.signals = FakeDetectorSignals()
        self.expandRatio = 2
        self.reset()
        self.missLimit = missLimit

    def reset(self):
        self.screenWidth, self.screenHeight = self.sc.getScreenSize()
        self.startedRecord = False
        self.startedDetect = False
        self.shots = []
        self.trackBoxes = []
        self.fakeScores = []
        self.fakeScoreTimestamp = []
        self.detectIndex = -1

    def screenRecording(self, monitor):
        while self.startedRecord:
            sctImg = self.sc.shot(monitor)
            self.shots.append(sctImg)
            # print("Frame appended",len(self.shots))

    def start(self, bbox):
        if self.startedRecord:
            print('Threaded Screen capturing is already started')
            return None
        # Computing screen monitor
        left, top, right, bottom = bbox[:-1]
        # print(int(left),int(top),int(right),int(bottom))
        center_x = (right + left) / 2
        center_y = (bottom + top) / 2
        width_exp = self.expandRatio * (right - left) / 2
        height_exp = self.expandRatio * (bottom - top) / 2

        monitorleft = max(int(center_x - width_exp), 0)
        monitortop = max(int(center_y - height_exp), 0)
        monitorright = min(int(center_x + width_exp), self.screenWidth)
        monitorbottom = min(int(center_y + height_exp), self.screenHeight)
        monitor = {"top": monitortop, "left": monitorleft, "width": monitorright - monitorleft, "height": monitorbottom - monitortop}

        # print(monitor)

        # starting a thread for screen capturing
        self.startedRecord = True
        self.captureThread = threading.Thread(target=self.screenRecording, args=(monitor,))

        self.captureThread.start()
        #face detection thread
        shiftBbox = [left - monitorleft, top - monitortop, right - monitorleft, bottom - monitortop, bbox[-1]]# change in the starting point, Shift the box
        # print(shiftBbox)
        self.startedDetect=True
        self.faceTrackingThread = threading.Thread(target=self.faceTracking, args=(shiftBbox,))

        self.faceTrackingThread.start()
        return self

    def faceTracking(self, prvBbox):
        """

        :param prvBbox:[left,top,right,bottom,score]
        :return:
        """
        winlength= self.fakeDetector.winlength
        overlap=self.fakeDetector.overlap
        faceChunk = Queue()
        frameNumbers=Queue()

        cleanIndex = 0
        missCount = 0
        # print(prvBbox)
        while (self.startedRecord or self.detectIndex + 1 < len(self.shots)) and self.startedDetect:
            if self.detectIndex + 1 < len(self.shots):
                self.detectIndex += 1
                shot = self.shots[self.detectIndex]
                image = self.sc.processScreenshot(shot)

                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                bboxes = self.faceDetector.detect(image)
                # print(bboxes)
                if len(bboxes)!=0:
                    bbox = self.faceDetector.selectBox(bboxes, prvBbox, image)
                    if bbox is not None:
                        #add box to list
                        self.trackBoxes.append(bbox)
                        faceChunk.put(self.faceDetector.cropBox(bbox[:-1],image))
                        frameNumbers.put(self.detectIndex+1)

                        prvBbox=bbox
                        missCount=0
                    else:
                        print("Not matched with previous box")
                        self.trackBoxes.append(None)
                        #box not mathced with previous, increase miss count
                        missCount += 1

                else:
                    print("No detection")
                    self.trackBoxes.append(None)
                    missCount+= 1
                # data processed signal
                self.signals.dataProcessedUpdate.emit((self.detectIndex+1,len(self.shots)))
                # print("Face Box appended", len(self.trackBoxes))
                # print(self.trackBoxes[-1])

            if missCount>=self.missLimit:
                print("Face miss limit crossed")
                self.stopDetection()

            #fake detection on the image chunk
            if faceChunk.qsize()== winlength:
                #create a thread for fake detection
                fakeDetectionThread = threading.Thread(target=self.fakeDetection, args=(list(faceChunk.queue),list(frameNumbers.queue)),)
                fakeDetectionThread.start()
                #clear chunk
                for i in range(winlength-overlap):
                    faceChunk.get()
                    number=frameNumbers.get()
                #remove previous shots
                self.shots[cleanIndex:number] = [None for i in range(number-cleanIndex)]
                cleanIndex=number





    def fakeDetection(self,faceChunk,frameNumbers):
        """

        :param faceChunk: list of rgb numpy images
        :param frameNumbers: list of frame numbers corresponding to the face images
        :return:
        """
        fakescore = self.fakeDetector.predict([faceChunk])[0]
        self.fakeScores.append(fakescore)
        middleTimestamp = frameNumbers[int(self.fakeDetector.winlength/2)]
        self.fakeScoreTimestamp.append(middleTimestamp)
        self.signals.scoreUpdate.emit((middleTimestamp, fakescore))

    def stopRecording(self):
        self.startedRecord = False

    def stopDetection(self):
        self.stopRecording()
        self.startedDetect = False





