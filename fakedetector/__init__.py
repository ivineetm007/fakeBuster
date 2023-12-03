

class FakeDetector(object):
    def __init__(self,device='cpu',imgdim= 224, winLength=30, overlap=0, fakeclass = 0):
        self.imgdim = imgdim
        self.winlength = winLength
        self.overlap = overlap
        self.device = device
        self.fakeclass = fakeclass
        self.model = None

    def load_model(self, path):
        pass

    def predict(self, chunkslist):
        """
            :param chunkslist: list of chunks [[frames]]
            :return: list of fake scores for each chunk, list length = number of chunks/ len(chunkslist)
        """
        pass


