#UI parameters
#app
iconsPath='ui/icons'
iconsList = [('32x32.png', 32),('64x64.png', 64), ('128x128.png', 128), ('256x256.png', 256)]
#splash
splashPath= 'ui/splash2.png'
splashWidth= 600
splashHeight= 300
##Main Window
windowTitle = 'FakeBuster AI'
plotColors=[
            (0, 255, 0),
            (255,165,0),
            (255, 0, 0)
        ]
##Dialog
dialogTitle = "Fake score summary"
dialogImageWidth = 141
dialogImageHeight = 141
# windowWidth = 400
# windowHeight =
faceIconSize = 110
#face detector parameters
weights_path = 'checkpoints\sfd_face.pth'
#fake detector parameters
checkpointpath = 'checkpoints\model_best_epoch20.pth.tar'#one step training
# checkpointpath = 'checkpoints\epoch50.pth.tar'#two step training
fake_det_device = 'cuda'# ['cuda' or 'cpu']
face_det_device = 'cuda'# ['cuda' or 'cpu']
skip = 1 # 1 by default
winlength = 30
overlap = 12
