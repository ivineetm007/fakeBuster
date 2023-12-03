import threading
from datetime import datetime
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QObject


class FakeDetectorSignals(QObject):
    """
    Defines the signals available from a running thread.

    Supported signals are:
    scoreUpdate
        `tuple` (timestamp,score)
    dataProcessedUpdate
        `tuple` (processed count,captured count)
    statusUpdate
        `str` status bar message
    """
    scoreUpdate = pyqtSignal(tuple)
    dataProcessedUpdate = pyqtSignal(tuple)
    statusUpdate = pyqtSignal(str)
    stopSignal = pyqtSignal(str)


class FakeDetectionThreading(object):
    def __init__(self, faceDetector, fakeDetector, screenCapture, missLimit=20, skip=1):
        self.faceDetector = faceDetector
        self.fakeDetector = fakeDetector
        self.sc = screenCapture
        self.signals = FakeDetectorSignals()
        self.expandRatio = 2
        self.reset()
        self.missLimit = missLimit
        self.skip = skip


    def reset(self):
        self.screenWidth, self.screenHeight = self.sc.getScreenSize()
        self.startedRecord = False
        self.startedDetect = False
        self.shots = []
        self.systemTimeList = []
        self.trackBoxes = []
        self.fakeScores = []
        self.fakeScoreTimestamp = []
        self.detectIndex = -1

    def screenRecording(self, monitor):
        while self.startedRecord:
            sctImg = self.sc.shot(monitor)
            self.shots.append(sctImg)
            self.systemTimeList.append(datetime.now().strftime('%H:%M:%S.%f')[:-4])

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
        #face tracking thread
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
        frameNumbers = Queue()

        cleanIndex = 0
        missCount = 0
        chunkcount = 0
        # print(prvBbox)
        while (self.startedRecord or self.detectIndex + 1 < len(self.shots)) and self.startedDetect:
            if self.detectIndex + 1 < len(self.shots):
                self.detectIndex += 1
                shot = self.shots[self.detectIndex]
                image = self.sc.processScreenshot(shot)
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
                        self.trackBoxes.append(None)
                        #box not mathced with previous, increase miss count
                        missCount += 1

                else:
                    self.trackBoxes.append(None)
                    missCount += 1

                # data processed signal
                self.signals.dataProcessedUpdate.emit((self.detectIndex + 1, len(self.shots)))
            if missCount >= self.missLimit:
                self.signals.stopSignal.emit("Face not matched or moved out of place. Fake detection stopped")


            #fake detection on the image chunk
            if faceChunk.qsize()== winlength:

                #create a thread for fake detection
                if chunkcount % self.skip == 0:
                    fakeDetectionThread = threading.Thread(target=self.fakeDetection, args=(list(faceChunk.queue),list(frameNumbers.queue)),)
                    fakeDetectionThread.start()
                chunkcount += 1
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








