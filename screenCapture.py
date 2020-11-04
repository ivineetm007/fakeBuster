import numpy as np
import cv2
import mss.tools


class MSSscreenCaptue(object):
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.getMonitor()
        self.width = self.monitor['width']
        self.height = self.monitor['height']


    def getMonitor(self, index=1):
        return self.sct.monitors[index]

    def processScreenshot(self, screenshot):
        img = np.array(screenshot)
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imgrgb

    def shot(self, monitor):
        return self.sct.grab(monitor)

    def fullScreenshot(self):
        screenshot = self.sct.grab(self.monitor)
        return self.processScreenshot(screenshot)

    def getScreenSize(self):
        return (self.width, self.height)


