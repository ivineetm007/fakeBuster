import os
from math import ceil

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QSize
from screenCapture import MSSscreenCaptue
from fakedetector import FakeDetectionThreading, MDSFakeDetector
from models import FaceDetector
import pyqtgraph as pg
import numpy as np
import time
import sys
import config
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QListView

qt_creator_file = "ui/mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)


class FaceListModel(QtCore.QAbstractListModel):
    '''
        Model to handle faces detected on screen
    '''

    def __init__(self, *args, **kwargs):
        super(FaceListModel, self).__init__(*args, **kwargs)
        self.faceWidgets = []

    def data(self, index, role):
        if role == Qt.DecorationRole:
            _,widget = self.faceWidgets[index.row()]
            return widget

    def rowCount(self, index):
        return len(self.faceWidgets)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, fakedetector, facedetector ,width=400, faceIconSize=140):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.windowWidth = width
        self.faceIconSize = faceIconSize
        self.setupUi(self)  # load ui
        self.faceModel = FaceListModel()
        self.faceView.setModel(self.faceModel)
        self.faceView.setViewMode(QListView.IconMode)
        self.faceView.setResizeMode(QListView.Adjust)
        self.iconQsize=QSize(self.faceIconSize, self.faceIconSize)
        self.faceView.setIconSize(self.iconQsize)
        self.faceView.setSpacing(2)
        #Buttons slots
        self.scanButton.pressed.connect(self.scan)
        self.startRecordButton.pressed.connect(self.startTracking)
        self.stopRecordButton.pressed.connect(self.stopRecording)
        self.stopDetectButton.pressed.connect(self.stopTracking)


        # setting  the fixed width of window
        # self.setFixedWidth(self.windowWidth)
        #screen capture

        self.sc=MSSscreenCaptue()

        #face detector
        self.faceDetector = facedetector
        self.fakeDetector = fakedetector
        self.fakeTracker = FakeDetectionThreading(faceDetector=self.faceDetector, fakeDetector=self.fakeDetector, screenCapture=self.sc)
        self.fakeTracker.signals.scoreUpdate.connect(self.updatefakescore)
        self.fakeTracker.signals.dataProcessedUpdate.connect(self.updatedatastats)
        #plotting
        self.initializePlot()



    def scan(self):
        #clear scanned faces
        self.faceModel.faceWidgets = []
        self.faceModel.layoutChanged.emit()
        self.statusBar().showMessage("Detecting Faces.........")
        faceModelData=[]
        #take screen shot
        screenImage=self.sc.fullScreenshot()
        #detect faces
        bboxes=self.faceDetector.detect(screenImage)
        if len(bboxes!=0):
            #crop faces
            for bbox in bboxes:
                faceImg=self.faceDetector.cropBox(box=bbox[:-1],image=screenImage)
                faceImg = np.require(faceImg, np.uint8, 'C')
                height, width, channel = faceImg.shape  # rgb order
                bytesPerLine = 3 * width
                # print(faceImg.shape)
                # print(faceImg.dtype)
                # print(faceImg.data)
                qImg = QImage(faceImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixImg = QtGui.QPixmap(qImg)
                icon = QtGui.QIcon(pixImg.scaled(self.faceIconSize,self.faceIconSize))
                faceModelData.append([bbox,icon])
            self.statusBar().showMessage(str(len(bboxes))+" Faces detected")
        else:
            # setting status bar message
            self.statusBar().showMessage("No Face detected")

        self.faceModel.faceWidgets = faceModelData
        self.faceModel.layoutChanged.emit()

        # cv2.imshow(faceImg)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', faceImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def startTracking(self):
        indexes = self.faceView.selectedIndexes()
        if indexes:
            index = indexes[0]
            bbox,_=self.faceModel.faceWidgets[index.row()]
            #reset
            self.fakeTracker.reset()
            self.resetPlot()
            self.updatedatastats((0,0,))
            # start
            self.fakeTracker.start(bbox)
            self.statusBar().showMessage("Face Tracking and Fake Detection started.....")
            # print()
        else:
            self.statusBar().showMessage("No Face selected.")

    def stopRecording(self):
        self.statusBar().showMessage("Stopping Screen Recording....")
        self.fakeTracker.stopRecording()

    def stopTracking(self):
        self.statusBar().showMessage("Stopping Fake Detection....")
        self.fakeTracker.stopDetection()

    def initializePlot(self):
        self.fakeScores = []
        self.fakeTimestamps = []
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setYRange(0, 1.2, padding=0)
        self.dataLine = pg.PlotDataItem(self.fakeTimestamps, self.fakeScores, symbol='o')
        self.graphWidget.addItem(self.dataLine)

    def resetPlot(self):
        self.fakeScores = []
        self.fakeTimestamps = []
        self.dataLine.setData(self.fakeTimestamps, self.fakeScores)

    def updatedatastats(self, signaldata):
        processed, captured = signaldata
        self.capturedView.setText(str(captured))
        self.processedView.setText(str(processed))


    def updatefakescore(self,signaldata):
        timestamp, score = signaldata
        self.fakeScores.append(score)
        self.fakeTimestamps.append(timestamp)
        self.dataLine.setData(self.fakeTimestamps, self.fakeScores)





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #loading fake detector
    device='cuda'
    facedetector=FaceDetector(device=device)
    fakedetector = MDSFakeDetector(device=device, overlap=config.overlap, winLength=config.winlength)
    fakedetector.loadcheckpoint(config.checkpointpath)
    window = MainWindow(fakedetector=fakedetector,facedetector=facedetector)
    window.show()
    app.exec_()
