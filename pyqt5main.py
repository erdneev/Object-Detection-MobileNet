import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog,QLabel,QAction,QMainWindow,QApplication
import cv2
from detection_model import ObjectDetectionMobileNetModel

class Window(QMainWindow):
    resized = QtCore.pyqtSignal()
    def __init__(self):
        super(Window, self).__init__()
        
        model = ObjectDetectionMobileNetModel("./MobileNetSSD_deploy.prototxt.txt", "./MobileNetSSD_deploy.caffemodel", 0.4)
        self.model = model

        self.setGeometry(100, 100, 500, 500)
        self.setWindowTitle("MobileNet Object Detection Showcase")

        openFile = QAction("&File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip("Open File")
        openFile.triggered.connect(self.fileOpen)

        self.statusBar()

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(openFile)

        self.lbl = QLabel(self)
        self.setCentralWidget(self.lbl)
        self.resized.connect(self.resizePixmap)
        self.setCurrentPixmap("./test_images/aeroplane.jpg")

        self.home()

    def home(self):
        self.show()

    def setCurrentPixmap(self, path):
        image = cv2.imread(path)
        image = self.model.process(image)
        qImage = QtGui.QImage(image, image.shape[1],\
                            image.shape[0], image.shape[1] * 3,QtGui.QImage.Format_RGB888).rgbSwapped()
        self.currentImagePixmap = QtGui.QPixmap(qImage)

    def resizeEvent(self, event):
        self.resized.emit()
        return super(Window, self).resizeEvent(event)

    def fileOpen(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        self.setCurrentPixmap(name[0])
        # pixmap = QtGui.QPixmap(name[0])
        self.resizePixmap()

    def resizePixmap(self):
        self.lbl.setPixmap(self.currentImagePixmap.scaled(self.lbl.size()))


def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()