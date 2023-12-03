from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QSize
from screenCapture import MSSscreenCaptue
from screenfakedetector import FakeDetectionThreading
import pyqtgraph as pg
import numpy as np
import sys
from screenfacedetector.detectors import S3FDDetector
import config
from fakedetector.MDS import MDSFakeDetector
import bisect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QListView, QLabel, QDialog,QSplashScreen
import utils

qt_creator_file = "ui/mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_creator_file)
dialog_file = "ui/analysis_dialog.ui"
Ui_Dialog, _ = uic.loadUiType(dialog_file)


class FaceListModel(QtCore.QAbstractListModel):
    '''
        Model to handle faces detected on screen
    '''

    def __init__(self, *args, **kwargs):
        super(FaceListModel, self).__init__(*args, **kwargs)
        self.faceWidgets = []

    def data(self, index, role):
        if role == Qt.DecorationRole:
            _,widget,_ = self.faceWidgets[index.row()]
            return widget

    def rowCount(self, index):
        return len(self.faceWidgets)

class AnalysisDlg(QDialog):
    """Analysis dialog."""
    def __init__(self, parent=None, pixmap=None, **kwargs):
        super().__init__(parent,**kwargs)
        # Create an instance of the GUI
        self.ui = Ui_Dialog()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
        if pixmap != None:
            self.ui.imageView.setPixmap(pixmap)
    def showStats(self,maxLabel,maxValue,minLabel,minValue,avgLabel,avgValue):
        self.ui.maxLabel.setText(maxLabel)
        self.ui.maxValue.setText(maxValue)
        self.ui.minLabel.setText(minLabel)
        self.ui.minValue.setText(minValue)
        self.ui.avgLabel.setText(avgLabel)
        avgLabel = "HIGH"
        self.ui.avgValue.setStyleSheet("color: red")
        if avgValue < 0.33:
            avgLabel = "LOW"
            self.ui.avgValue.setStyleSheet("color: green")
        elif avgValue < 0.66:
            avgLabel = "MEDIUM"
            self.ui.avgValue.setStyleSheet("color: orange")
        self.ui.avgValue.setText(avgLabel)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, fakedetector, facedetector):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        # self.windowWidth = width
        self.faceIconSize = config.faceIconSize
        self.setupUi(self)  # load ui
        # window settings
        self.setWindowTitle(config.windowTitle)
        # Face Icons view and model
        self.faceModel = FaceListModel()
        self.faceView.setModel(self.faceModel)
        self.faceView.setViewMode(QListView.IconMode)
        self.faceView.setResizeMode(QListView.Adjust)
        self.iconQsize = QSize(self.faceIconSize, self.faceIconSize)
        self.faceView.setIconSize(self.iconQsize)
        self.faceView.setSpacing(2)
        # Buttons slots
        self.scanButton.pressed.connect(self.scan)
        self.startButton.pressed.connect(self.startTracking)
        self.stopButton.pressed.connect(self.stopTracking)
        self.statsButton.pressed.connect(self.showAnalysis)
        #set state of buttons
        self.stopButton.setEnabled(False)

        # Screen capture
        self.sc = MSSscreenCaptue()
        # Face detector
        self.faceDetector = facedetector
        self.fakeDetector = fakedetector
        self.fakeTracker = FakeDetectionThreading(faceDetector=self.faceDetector, fakeDetector=self.fakeDetector, screenCapture=self.sc, skip=config.skip)
        self.fakeTracker.signals.scoreUpdate.connect(self.updatefakescore)
        self.fakeTracker.signals.statusUpdate.connect(self.showMessage)
        self.fakeTracker.signals.dataProcessedUpdate.connect(self.updatedatastats)
        self.fakeTracker.signals.stopSignal.connect(self.stopTracking)
        #Plotting
        self.initializePlot()

        #fisrt scan
        self.scan(firstScan=True)

        #changing status bar
        # creating a label widget
        self.processedLabel = QLabel("Processed")
        font = QtGui.QFont()
        font.setBold(True)
        self.processedLabel.setFont(font)
        self.processedView = QLabel("0/0")
        # adding labels to status bar
        self.statusBar().addPermanentWidget(self.processedLabel)
        self.statusBar().addPermanentWidget(self.processedView)

    def showMessage(self,msg):
        self.statusBar().showMessage(msg)

    def scan(self, firstScan = False):
        #clear scanned faces
        self.faceModel.faceWidgets = []
        self.faceModel.layoutChanged.emit()
        self.showMessage("Detecting Faces.........")
        faceModelData=[]
        # take screen shot
        screenImage=self.sc.fullScreenshot()
        # Mainwindow bounding box- [left, top, right, bottom]
        windowBox=[self.geometry().x(), self.geometry().y(), self.geometry().x()+self.width(), self.geometry().y()+self.height()]
        # detect faces
        bboxes=self.faceDetector.detect(screenImage)
        if len(bboxes!=0):
            #crop faces
            count = 0
            for bbox in bboxes:
                if (not firstScan) and utils.check_inside_box(windowBox,bbox[:-1]):
                    continue
                faceImg=self.faceDetector.cropBox(box=bbox[:-1],image=screenImage, crop_scale=0.2)
                faceImg = np.require(faceImg, np.uint8, 'C')
                height, width, channel = faceImg.shape  # rgb order
                bytesPerLine = 3 * width

                qImg = QImage(faceImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixImg = QtGui.QPixmap(qImg)
                icon = QtGui.QIcon(pixImg.scaled(self.faceIconSize, self.faceIconSize))
                faceModelData.append([bbox,icon,pixImg])
                count += 1
            self.showMessage(str(count)+" faces detected")
        else:
            # setting status bar message
            self.showMessage("No face detected")

        self.faceView.clearSelection()
        self.faceModel.faceWidgets = faceModelData
        self.faceModel.layoutChanged.emit()


    def startTracking(self):
        indexes = self.faceView.selectedIndexes()
        if indexes:
            self.showMessage("Starting fake detection..........")
            self.selectedIndex = indexes[0]
            bbox,_, Qimg=self.faceModel.faceWidgets[self.selectedIndex.row()]
            #reset
            self.fakeTracker.reset()
            self.resetPlot()
            self.updatedatastats((0,0))
            # start
            self.fakeTracker.start(bbox)
            self.showMessage("Face tracking and fake detection started.")
            self.startButton.setEnabled(False)
            self.stopButton.setEnabled(True)
        else:
            self.showMessage("No face selected.")

    def stopTracking(self, msg=None):
        self.showMessage("Stopping fake detection....")
        self.fakeTracker.stopDetection()
        if msg == None:
            self.showMessage("Fake detection stopped.")
        else:
            self.showMessage(msg)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def initializePlot(self):
        self.fakeScores = []
        self.fakeTimestamps = []
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setYRange(0, 1.2, padding=0)
        self.plotColors = config.plotColors

        # color map
        self.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(self.plotColors)), color=self.plotColors)
        self.brushes = [pg.mkBrush(self.cmap.map(score)) for score in self.fakeScores]
        self.dataLine = pg.PlotDataItem(self.fakeTimestamps, self.fakeScores, symbol='o', symbolBrush=self.brushes)

        self.graphWidget.addItem(self.dataLine)

    def resetPlot(self):
        self.fakeScores = []
        self.fakeTimestamps = []
        self.brushes = []
        self.dataLine.setData(self.fakeTimestamps, self.fakeScores, symbolBrush=self.brushes)

    def updatedatastats(self, signaldata):
        processed, captured = signaldata
        self.processedView.setText(str(processed) + '/' + str(captured))

    def updatefakescore(self, signaldata):
        timestamp, score = signaldata
        brush = pg.mkBrush(self.cmap.map(score))
        ind = bisect.bisect(self.fakeTimestamps, timestamp)
        self.fakeScores.insert(ind, score)
        self.fakeTimestamps.insert(ind, timestamp)
        self.brushes.insert(ind, brush)
        self.dataLine.setData(self.fakeTimestamps, self.fakeScores, symbolBrush=self.brushes)

    def showAnalysis(self):
        if len(self.fakeScores)!=0:
            maxIndex = np.argmax(self.fakeScores)
            # maxTimestamp = self.fakeTimestamps[maxIndex]
            maxTimestamp = self.fakeTracker.systemTimeList[maxIndex]
            maxLabel="Most manipulated at {}".format(maxTimestamp)
            maxScore = round(self.fakeScores[maxIndex], 2)
            minIndex = np.argmin(self.fakeScores)
            minScore = round(self.fakeScores[minIndex], 2)
            # minTimestamp = self.fakeTimestamps[minIndex]
            minTimestamp = self.fakeTracker.systemTimeList[minIndex]
            minLabel = "Least manipulated at {}".format(minTimestamp)
            avgLabel = "Average imposter Score"
            avgScore = round(np.mean(self.fakeScores), 2)
            _, icon, pixmap = self.faceModel.faceWidgets[self.selectedIndex.row()]
            dlg = AnalysisDlg(parent=self, pixmap=pixmap)
            dlg.showStats(maxLabel=maxLabel, maxValue=str(maxScore), minLabel=minLabel, minValue=str(minScore), avgLabel=avgLabel, avgValue=avgScore)
        else:
            dlg = AnalysisDlg(parent=self)

        dlg.setWindowTitle(config.dialogTitle)
        # msg.setText(msgstr)
        returnValue = dlg.exec()
        # returnValue = msg.exec()

    def closeEvent(self, *args, **kwargs):
        self.stopTracking()
        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # set app icon
    app_icon = QtGui.QIcon()
    for filename,size in config.iconsList:
        app_icon.addFile(config.iconsPath+'/'+filename,QtCore.QSize(size, size))
    app.setWindowIcon(app_icon)

    splash_pix = QPixmap(config.splashPath)#.scaled(config.splashWidth, config.splashHeight)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()

    # loading face detector
    splash.showMessage("Loading Face detector.............", alignment=Qt.AlignBottom)
    facedetector = S3FDDetector(config.weights_path, device=config.face_det_device)
    #loading fake detector
    splash.showMessage("Loading Fake detector.............", alignment=Qt.AlignBottom)
    fakedetector = MDSFakeDetector(device=config.fake_det_device, overlap=config.overlap, winLength=config.winlength)
    fakedetector.load_model(config.checkpointpath)


    window = MainWindow(fakedetector=fakedetector, facedetector=facedetector)
    splash.finish(window)
    window.show()
    app.exec_()
