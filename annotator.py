# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:19:53 2022

@author: lowes
"""


import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui
import numpy as np
import random
from PIL import Image
# from config import (get_args, update_args)
# from model_unet import (AbstractUNet, AbstractDownNet)
from PIL.ImageQt import ImageQt 
import qimage2ndarray
import os
import glob
import torch
import requests
import cv2
import imutils
import SimpleITK as sitk
from scipy.ndimage import zoom
import utils


  
class Annotator(PyQt5.QtWidgets.QWidget):
    
    def __init__(self, size=None, ct_shape=[10,512,512], resize_size = 128):
        '''
        Initializes an Annotator without the image.
        Parameters
        ----------
        size : two-element tupple for the size of the annotator.
        '''
        
        super().__init__() 
        
        
        if size is None:
            size = PyQt5.QtCore.QSize(256,256)
        elif type(size) is tuple:
            size = PyQt5.QtCore.QSize(size[0],size[1])
            
        # Pixmap layers
        self.imagePix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        self.imagePix.fill(self.color_picker(label=0, opacity=0))
        self.annotationPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), 
                                                 self.imagePix.height())
        self.annotationPix.fill(self.color_picker(label=0, opacity=0))
        
        self.resizePix = PyQt5.QtGui.QPixmap(ct_shape[2], ct_shape[1])
        self.resizePix.fill(self.color_picker(label=0, opacity=0))
        
        
        self.cursorPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), 
                                             self.imagePix.height())
        self.cursorPix.fill(self.color_picker(label=0, opacity=0))  
        self.masks_pix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        self.masks_pix.fill(self.color_picker(label=0, opacity=0))
        self.size = size
        # Atributes for drawing
        self.ct_shape = ct_shape
        self.resize_size = resize_size
        self.pen_resize = 8
        self.label = 1
        self.penWidth = int(size.width()/40)
        self.lastDrawPoint = PyQt5.QtCore.QPoint()
        
        # Atributes for displaying
        self.overlay = 0
        self.overlays = {0:'both', 1:'annotation', 2:'image'}
        self.annotationOpacity = 0.4
        self.cursorOpacity = 0.5
        self.zoomOpacity = 0.5
        self.setTitle()
        self.setCursor(PyQt5.QtGui.QCursor(PyQt5.QtCore.Qt.CrossCursor))
        self.lastCursorPoint = PyQt5.QtCore.QPoint()
        self.setMouseTracking(True)
        
        # Atributes relating to the transformation between widget 
        # coordinate system and image coordinate system
        self.zoomFactor = 1 # accounts for resizing of the widget and for zooming in the part of the image
        self.padding = PyQt5.QtCore.QPoint(0, 0) # padding when aspect ratio of image and widget does not match
        #TODO changed?
        self.target = PyQt5.QtCore.QRect(0, 0, size.width(),size.height()) #self.width(),self.height()) # part of the target being drawn on
        self.source = PyQt5.QtCore.QRect(0, 0, 
                self.imagePix.width(), self.imagePix.height()) # part of the image being drawn
        self.offset = PyQt5.QtCore.QPoint(0, 0) # offset between image center and area of interest center
        
        # Flags needed to keep track of different states
        self.zPressed = False # when z is pressed zooming can start
        self.activelyDrawing = False
        self.newZoomValues = None
        self.done_predicting = True
        
        # Label for displaying text overlay
        self.textField = PyQt5.QtWidgets.QLabel(self)
        self.textField.setStyleSheet("background-color: rgba(191,191,191,191)")
        self.textField.setTextFormat(PyQt5.QtCore.Qt.RichText)
        self.textField.resize(0,0)
        self.textField.move(10,10)     
        self.hPressed = False
        self.textField.setAttribute(PyQt5.QtCore.Qt.WA_TransparentForMouseEvents)
        #self.textField.setAttribute(PyQt5.QtCore.Qt.WA_TranslucentBackground) # considered making bacground translucent      
        
        # Timer for displaying text overlay
        self.timer = PyQt5.QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hideText)
        
        self.mouseRelease = False
        # Filename for saving annotations
        self.saveAddress = 'annotations.png'

        # Playtime
        initial_zoom = min(2000/max(self.imagePix.width(), 
                4*self.imagePix.height()/3),1) # downsize if larger than (2000,1500)
        self.resize(initial_zoom*self.imagePix.width(), 
                    initial_zoom*self.imagePix.height())
        # self.show() - moved out to wher Annotator is called
        
        # self.oldAnn,self.oldResize = [], []
        self.oldAnn,self.oldResize = [self.annotationPix.copy()], [self.resizePix.copy()]
        
        self.showInfo(self.introText(),5000)
        # self.showMaximized()
        print(self.introText(False))
    
    @classmethod
    def fromFolder(cls, folder_name, net_name, resize_size, resize_shape, use_dicom = False):
        print("folder init")
        
        if use_dicom:
            ct_list = glob.glob(os.path.join(folder_name,"data","PANCREAS_**"))
            name = ct_list[0]
            ct_vol = utils.read_dicom_files_carefully(name)
            spacing = [1,1,1]
            
        else:
            
            ct_list = glob.glob(os.path.join(folder_name,"*.nii.gz"))
            # random.shuffle(ct_list)
            name = ct_list.pop(2)
            sitk_t1 = sitk.ReadImage(name)
            spacing = sitk_t1.GetSpacing()
            ct_vol = sitk.GetArrayFromImage(sitk_t1)
        
        ct_vol[ct_vol<-1024] = -1024
        ct_vol[ct_vol>1024] = 1024
        ct_vol = (255*(ct_vol - np.min(ct_vol))/np.ptp(ct_vol)).astype(np.uint8)
        
        if resize_shape is None:
            resize_shape = (ct_vol.shape[2],ct_vol.shape[1])
            
        
        im = cv2.resize(ct_vol[0,:,:],resize_shape)
        gray = im.copy() # check whether needed
        
        bytesPerLine = gray.nbytes//gray.shape[0]
        qimage = PyQt5.QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                    bytesPerLine,
                                    PyQt5.QtGui.QImage.Format_Grayscale8)
        imagePix = PyQt5.QtGui.QPixmap(qimage)
        # im1 = im.copy() # check whether needed
        # # im = np.require(im, np.uint8, 'C')
        # # bytesPerLine = im.shape[0]*3
        # totalBytes = im1.nbytes
        # # divide by the number of rows
        # bytesPerLine = int(totalBytes/im1.shape[0])
        
        # qimage = PyQt5.QtGui.QImage(im1.data, im1.shape[1], im1.shape[0],bytesPerLine,
        #                             PyQt5.QtGui.QImage.Format_RGB888)
        imagePix = PyQt5.QtGui.QPixmap(qimage)
        annotator = Annotator(imagePix.size(), ct_vol.shape, resize_size)
        annotator.imagePix = imagePix
        annotator.annotationsFilename = os.path.join("data/labels",os.path.basename(name))
        
        #TODO - fix network
        """ For using segmentation network
        device = "cuda"
        path = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/General_Interactive_Segmentation/runs"
        name = os.path.join(path, net_name+".pt")
        
        # arg_name = ''.join(filter(lambda x: not x.isdigit(),net_name))
        arg_name = net_name.split('_')[0]
        args = get_args(name=arg_name)
        annotator.recon_mode = args.training.recon_mode
        # args = get_args(name=arg_name[:-1])


        net = AbstractUNet(args).to(device)
      


        ckpt = torch.load(name, map_location=lambda storage, loc: storage)
        #args = ckpt['args']
        #args = update_args(args)
        net.load_state_dict(ckpt["net"])
        annotator.net = net
        annotator.net.eval()
        annotator.device = device
        """
        print("CT shape",ct_vol.shape)
        print("im shape",im.shape)
        annotator.ct_list = ct_list
        annotator.filename = name
        annotator.ct_vol = ct_vol
        annotator.resize_shape = resize_shape
        annotator.spacing=spacing
        annotator.cur_slice = 0  
        annotator.ct_list = ct_list
        annotator.use_dicom = use_dicom
        annotator.calculateSphere()
        annotator.predict() 
        annotator.show_slice()
        return annotator                            
    


    
    helpText = (
        '<i>Help for annotator</i> <br>' 
        '<b>KEYBOARD COMMANDS:</b> <br>' 
        '&nbsp; &nbsp; <b>1</b> to <b>9</b> changes pen label (L) <br>' 
        '&nbsp; &nbsp; <b>0</b> eraser mode <br>' 
        '&nbsp; &nbsp; <b>&uarr;</b> and <b>&darr;</b> changes pen width (W) <br>' 
        '&nbsp; &nbsp; <b>Enter</b> predicts mask from skelet <br>' 
        '&nbsp; &nbsp; <b>Z</b> undo last pencil brush <br>' 
        '&nbsp; &nbsp; <b>R</b> resets current image <br>' 
        '&nbsp; &nbsp; <b>I</b> loads a new image into the annotator <br>' 
        # '&nbsp; &nbsp; <b>S</b> saves annotation <br>' 
        '&nbsp; &nbsp; <b>H</b> shows this help <br>' 
        '<b>MOUSE DRAG:</b> <br>' 
        '&nbsp; &nbsp; Draws annotation <br>')
    
    @classmethod
    def introText(cls, rich = True):
        if rich:
            return '<i>Starting annotator</i> <br> For help, hit <b>H</b>'
            #'<hr> ANNOTATOR <br> Copyright (C) 2020 <br> Vedrana A. Dahl'
        else:
            return "Starting annotator. For help, hit 'H'."

    def reset_masks(self):
        self.masks_pix = PyQt5.QtGui.QPixmap(self.size.width(), self.size.height()) 
        self.masks_pix.fill(self.color_picker(label=0, opacity=0))
        
    def reset_image(self):
        path = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/General_Interactive_Segmentation"

        if self.url is not None:
            # url = "http://192.168.0.112:8080/shot.jpg"
            while True:
                img_resp = requests.get(self.url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, -1)
                img = imutils.resize(img, width=1000, height=1800)
                cv2.imshow("Android_cam", img)
              
                # Press Esc key to exit
                if cv2.waitKey(1) != -1:
                    break
              
            cv2.destroyAllWindows()
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            if self.medical_images:
                im_list = glob.glob(os.path.join(path,"CHAOS_Train_Sets/Train_Sets/images","*.png"))
            else:
                im_list = glob.glob(os.path.join(path,'benchmark/dataset/img','*.jpg'))
            im_name = random.choice(im_list)
            print(os.path.basename(im_name))
            im = np.array(Image.open(im_name))
        
        if im.ndim==2:
            im = np.stack((im,im,im),axis=2)

        b,g,r = im[:,:,0], im[:,:,1], im[:,:,2]
        if (b==g).all() and (b==r).all():
            im1 = b.copy()
            bytesPerLine = im1.nbytes//im1.shape[0]
            qimage = PyQt5.QtGui.QImage(im1.data, im1.shape[1], im1.shape[0],
                                        bytesPerLine,
                                        PyQt5.QtGui.QImage.Format_Grayscale8)
        else:
            im1 = im.copy() # check whether needed
            # im = np.require(im, np.uint8, 'C')
            totalBytes = im1.nbytes
            # divide by the number of rows
            bytesPerLine = int(totalBytes/im1.shape[0])
            
            qimage = PyQt5.QtGui.QImage(im1.data, im1.shape[1], im1.shape[0],bytesPerLine,
                                        PyQt5.QtGui.QImage.Format_RGB888)
            
        imagePix = PyQt5.QtGui.QPixmap(qimage)
        size = imagePix.size()
        self.imagePix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        self.imagePix.fill(self.color_picker(label=0, opacity=0))
        self.annotationPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), 
                                                 self.imagePix.height())
        self.annotationPix.fill(self.color_picker(label=0, opacity=0))
        self.resizePix = PyQt5.QtGui.QPixmap(self.ct_shape[2], self.ct_shape[1])
        self.resizePix.fill(self.color_picker(label=0, opacity=0))
        self.cursorPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), 
                                             self.imagePix.height())
        self.cursorPix.fill(self.color_picker(label=0, opacity=0))  
        self.masks_pix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        self.masks_pix.fill(self.color_picker(label=0, opacity=0))
        self.size = size
        self.imagePix = imagePix
        self.annotationsFilename = 'test_annotations.png'
        
        self.oldAnn,self.oldResize = [self.annotationPix.copy()], [self.resizePix.copy()]
        
        # self.zoomFactor = 1 # accounts for resizing of the widget and for zooming in the part of the image
        #self.padding = PyQt5.QtCore.QPoint(0, 0) # padding when aspect ratio of image and widget does not match
        #self.target = PyQt5.QtCore.QRect(0, 0, self.width(),self.height()) # part of the target being drawn on
        #self.source = PyQt5.QtCore.QRect(0, 0, 
        #self.imagePix.width(), self.imagePix.height()) # part of the image being drawn
        #self.offset = PyQt5.QtCore.QPoint(0, 0) # offset between image center and area of interest center
        
        device = "cuda"
        path = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/General_Interactive_Segmentation"


        self.orig_im = im
        
        
        # self.net = net
        # self.net.eval()
        # self.device = device
        
    def showHelp(self):
        self.timer.stop()
        self.showText(self.helpText)
    
    def showInfo(self, text, time=1000):
        if not self.hPressed:
            self.timer.start(time)
            self.showText(text)
    
    def showText(self, text):
        self.textField.setText(text)
        #self.textField.resize(self.textField.fontMetrics().size(PyQt5.QtCore.Qt.TextExpandTabs, text))
        self.textField.adjustSize()
        self.update()
          
    def hideText(self):
        self.textField.resize(0,0)
        self.update()
        
    def setTitle(self):
        self.setWindowTitle(f'L:{self.label}, W:{self.penWidth}, '+
                            f'O:{self.overlays[self.overlay]}')
    
    def makePainter_resize(self, pixmap, color):
        """" Returns scribble painter operating on a given pixmap. """
        painter_scribble = PyQt5.QtGui.QPainter(pixmap)
        pw = self.pen_resize
        painter_scribble.setPen(PyQt5.QtGui.QPen(color, 
                    pw*self.zoomFactor, PyQt5.QtCore.Qt.SolidLine, 
                    PyQt5.QtCore.Qt.RoundCap, PyQt5.QtCore.Qt.RoundJoin))
        # painter_scribble.translate(-self.offset)
        painter_scribble.translate(-0.25,-0.25) # a compromise between odd and even pen width
        painter_scribble.scale(1/self.zoomFactor, 1/self.zoomFactor)
        # painter_scribble.translate(-self.padding)        
        painter_scribble.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_Source)
        return painter_scribble
    
    def makePainter(self, pixmap, color, resize_bool=False):
        """" Returns scribble painter operating on a given pixmap. """
        painter_scribble = PyQt5.QtGui.QPainter(pixmap)
        pw = self.pen_resize if resize_bool else self.penWidth
        painter_scribble.setPen(PyQt5.QtGui.QPen(color, 
                    pw*self.zoomFactor, PyQt5.QtCore.Qt.SolidLine, 
                    PyQt5.QtCore.Qt.RoundCap, PyQt5.QtCore.Qt.RoundJoin))
        painter_scribble.translate(-self.offset)
        painter_scribble.translate(-0.25,-0.25) # a compromise between odd and even pen width
        painter_scribble.scale(1/self.zoomFactor, 1/self.zoomFactor)
        painter_scribble.translate(-self.padding)        
        painter_scribble.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_Source)
        return painter_scribble

    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.overlay != 1: # overlay 0 or 2
            painter_display.drawPixmap(self.target, self.imagePix, self.source)
        if self.overlay != 2: # overlay 0 or 1
            painter_display.drawPixmap(self.target, self.annotationPix, 
                                       self.source)
        painter_display.drawPixmap(self.target, self.cursorPix, self.source)
        
        painter_display.drawPixmap(self.target, self.masks_pix, self.source)
        
    def drawCursorPoint(self, point):
        """Called when cursorPix needs update due to pen change or movement"""
        self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # transparent
        painter_scribble = self.makePainter(self.cursorPix, 
                    self.color_picker(self.label, self.cursorOpacity)) # the painter used for cursor
        painter_scribble.drawPoint(point)   
    
    def mousePressEvent(self, event):
        if event.button() == PyQt5.QtCore.Qt.LeftButton: 
            if self.zPressed: # initiate zooming and not drawing
                self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
                self.lastCursorPoint = event.pos()
                self.activelyZooming = True
                self.newZoomValues = 0 # for distinction between reset and cancel
            else: # initiate drawing
                painter_scribble = self.makePainter(self.annotationPix, 
                        self.color_picker(self.label, 
                            (self.label>0)*self.annotationOpacity)) # the painter used for drawing        
                painter_scribble.drawPoint(event.pos())
                
                painter_resize = self.makePainter_resize(self.resizePix, 
                        self.color_picker(self.label, 
                            (self.label>0)*self.annotationOpacity)) # the painter used for drawing 
                point = PyQt5.QtCore.QPoint((event.x()-self.padding.x())/self.size.width() * self.ct_shape[2]/self.zoomFactor,
                                    (event.y()-self.padding.y())/self.size.height() * self.ct_shape[1]/self.zoomFactor)
                painter_resize.drawPoint(point)
                
                self.last_resize_point = point
                self.lastDrawPoint = event.pos()   
                self.activelyDrawing = True
                self.update_pred_point(point)
                # print("event pos", event.pos())
                # print("point pos", point)
            self.update()
    
    def mouseMoveEvent(self, event):
           
        if self.activelyDrawing: 
            painter_scribble = self.makePainter(self.annotationPix, 
                    self.color_picker(self.label, 
                            (self.label>0)*self.annotationOpacity)) # the painter used for drawing        
            painter_scribble.drawLine(self.lastDrawPoint, event.pos())
            self.lastDrawPoint = event.pos()
            painter_resize = self.makePainter_resize(self.resizePix, 
                    self.color_picker(self.label, 
                        (self.label>0)*self.annotationOpacity)) # the painter used for drawing 
            point = PyQt5.QtCore.QPoint((event.x()-self.padding.x())/self.size.width() * self.ct_shape[2],
                                (event.y()-self.padding.y())/self.size.height() * self.ct_shape[1])
            painter_resize.drawLine(self.last_resize_point, point)
            self.last_resize_point = point
            self.update_pred_point(point)
        # just moving around
        self.drawCursorPoint(event.pos())
            
        self.lastCursorPoint = event.pos()      
        self.update()
    
    def mouseReleaseEvent1(self, event):  
        
        if self.activelyDrawing:
            self.oldAnn.append(self.annotationPix.copy())
            self.oldResize.append(self.resizePix.copy())
            self.activelyDrawing = False

    
    def wheelEvent(self,event):

        # self.update_pred()
        self.reset_current_image()
        scroll = np.sign(event.angleDelta().y())
        new_slice = self.cur_slice + scroll
        if 0 < new_slice < self.ct_shape[0]:      
            self.cur_slice = new_slice
        
        self.show_slice()
    
    def leaveEvent(self, event):
        """Removes curser when mouse leaves widget. """
        self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
        self.update()
            
    def resizeEvent(self, event):
        """ Triggered by resizing of the widget window. """
        self.adjustTarget()
                
    def adjustTarget(self):
        """ Computes padding needed such that aspect ratio of the image is correct. """
        self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
        self.update()   

        zoomWidth = self.width()/self.source.width()
        zoomHeight = self.height()/self.source.height() 
        
        # depending on aspect ratios, either pad up and down, or left and rigth
        if zoomWidth > zoomHeight:
            self.zoomFactor = zoomHeight
            self.padding = PyQt5.QtCore.QPoint(int((self.width() 
                            - self.source.width()*self.zoomFactor)/2), 0)
        else:
            self.zoomFactor = zoomWidth
            self.padding = PyQt5.QtCore.QPoint(0, int((self.height()
                            - self.source.height()*self.zoomFactor)/2))
            
        self.target = PyQt5.QtCore.QRect(self.padding, 
                            self.rect().bottomRight() - self.padding)
                   
    def executeZoom(self):
        """ Zooms to rectangle given by newZoomValues. """
        self.newZoomValues.translate(-self.padding)
        self.source = PyQt5.QtCore.QRect(self.newZoomValues.topLeft()/self.zoomFactor,
                self.newZoomValues.size()/self.zoomFactor)
        self.source.translate(-self.offset)
        self.source = self.source.intersected(self.imagePix.rect()) 
        self.showInfo('Zooming to ' + self.formatQRect(self.source))     
        self.offset = self.imagePix.rect().topLeft() - self.source.topLeft()
        self.adjustTarget()
        self.newZoomValues = None
    
    def resetZoom(self):
        """ Back to original zoom """
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(), 
                                         self.imagePix.height())
        self.showInfo('Reseting zoom to ' + self.formatQRect(self.source))        
        self.offset = PyQt5.QtCore.QPoint(0,0)
        self.adjustTarget()        
        self.newZoomValues = None
        
        
    def draw_mask(self): 
        self.reset_masks()
        # print("new_size", (self.target.width(),self.target.height()))

        mask = self.pred[self.cur_slice]
        mask = cv2.resize(mask.astype(np.uint8),
                           (self.target.width(),self.target.height()),
                           )#interpolation= cv2.INTER_LINEAR)#INTER_NEAREST_EXACT)
        colors = np.array([
            [0, 0, 0], # background, transparency is always drawn with black
            [255, 0, 0], # label 1
            [0, 191, 0], # label 2
            [0, 0, 255], # etc
            [255, 127, 0],
            [0, 255, 191],
            [127, 0, 255],
            [191, 255, 0],
            [0, 127, 255],
            [255, 64, 191]])*self.annotationOpacity
        col = colors[1]
        painter_scribble = self.makePainter(self.masks_pix, 
                PyQt5.QtGui.QColor("white"))
        painter_scribble.setOpacity(self.annotationOpacity)
        z_h = np.zeros((self.padding.y(),self.target.width()))
        z_w = np.zeros((self.target.height(), self.padding.x()))
        
        # masks = cv2.resize(masks,((self.rect().bottomRight() - 2*self.padding).x()+1,\
        #                           (self.rect().bottomRight() - self.padding).y()+1))
        mask = np.hstack((z_w,mask,z_w)) if self.padding.x() else mask 
        mask = np.vstack((z_h,mask,z_h)) if self.padding.y() else mask
        qimage = qimage2ndarray.array2qimage((mask==1)[:,:,np.newaxis]*col,normalize=True)
        imagePix = PyQt5.QtGui.QPixmap.fromImage(qimage)

        # imagePix.fill(self.color_picker(label=0, opacity=0))
        mask = imagePix.createMaskFromColor(PyQt5.QtGui.QColor("black"), PyQt5.QtCore.Qt.MaskInColor)

        painter_scribble.setClipRegion(PyQt5.QtGui.QRegion(mask))
        painter_scribble.drawImage(PyQt5.QtCore.QPoint(),qimage)
        # print("works??")

    def calculateSphere(self):
        X, Y, Z = np.meshgrid(np.arange(-self.ct_shape[2]//2,self.ct_shape[2]//2),
                              np.arange(-self.ct_shape[1]//2,self.ct_shape[1]//2),
                              np.arange(-self.ct_shape[0]//2,self.ct_shape[0]//2))
        
        radiusXY = self.penWidth*self.ct_shape[2]/self.resize_shape[0]*0.5
        radiusZ = self.penWidth*self.ct_shape[2]/self.resize_shape[0]*0.5 * self.spacing[0]/self.spacing[2]
        
        self.sphere = (np.power((X)/radiusXY,2)
                +np.power((Y)/radiusXY,2)
                +np.power((Z)/radiusZ,2)<=1)
        self.sphere_nonzero = self.sphere.nonzero()
        
    def show_slice(self):
        im = self.ct_vol[self.cur_slice]
        im = cv2.resize(im,self.resize_shape)

        gray = im.copy() # check whether needed
        
        bytesPerLine = gray.nbytes//gray.shape[0]
        qimage = PyQt5.QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                    bytesPerLine,
                                    PyQt5.QtGui.QImage.Format_Grayscale8)
        self.imagePix = PyQt5.QtGui.QPixmap(qimage)
        # size = imagePix.size()
        # self.imagePix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        # self.imagePix.fill(self.color_picker(label=0, opacity=0))
        self.draw_mask()
        # self.showVolumeTmp()
        self.update()
    
    def update_pred_point(self,event):
        # event is in (x,y) from origo top left - already reshaped to ct_shape?
        point = [event.x()-self.padding.x(),event.y()-self.padding.y()]
        # point = [event.x(),event.y()]
        """ for one circle on one slice
        point[0] *= self.ct_shape[2]/self.resize_shape[0]
        point[1] *= self.ct_shape[1]/self.resize_shape[1]
        center = (int(point[0]),int(point[1]))
        
        center = (event.x(),event.y())
        radius = round(self.penWidth*self.ct_shape[2]/self.resize_shape[0]*0.5)
        cv2.circle(self.pred[self.cur_slice],center,radius,self.label,-1)
        
        
        X, Y, Z = np.meshgrid(np.arange(self.ct_shape[2]),np.arange(self.ct_shape[1]),np.arange(self.ct_shape[0]))
        radiusXY = self.penWidth*self.ct_shape[2]/self.resize_shape[0]*0.5
        radiusZ = self.penWidth*self.ct_shape[2]/self.resize_shape[0]*0.5 * self.spacing[0]/self.spacing[2]
        import time
        t = time.time()
        sphere = (np.power((X-event.x())/radiusXY,2)
                +np.power((Y-event.y())/radiusXY,2)
                +np.power((Z-self.cur_slice)/radiusZ,2) <=1)
        # cv2.imshow("debug",sphere[:,:,self.cur_slice].astype(float))
        # self.pred[sphere.transpose(2,0,1)] = 1
        print("old elapsed time",time.time()-t)
        self.draw_mask()
        """
        
        # fast try
        
        shift = (event.x() - self.ct_shape[2]//2,event.y() - self.ct_shape[1]//2,self.cur_slice - self.ct_shape[0]//2)
        # print("shift",shift)
        # print("event",event.x(),event.y(),self.cur_slice)
        # cur_sphere = np.roll(self.sphere,shift=shift, axis=(0,1,2)).transpose(2,1,0)
        
        indices = (self.sphere_nonzero[2]+shift[2], self.sphere_nonzero[1]+shift[1], self.sphere_nonzero[0]+shift[0])
        
        # cv2.imshow("debug",sphere[:,:,self.cur_slice].astype(float))
        self.pred[indices] = 1
    
    def update_pred(self):
        annotations = qimage2ndarray.rgb_view(self.resizePix.toImage())
        
        self.pred[self.cur_slice][annotations[:,:,0]>0] = 1
        self.reset_current_image()
    
    def predict(self):
        
        #TODO - predict with network
        if self.use_dicom:
            file_num = ''.join([s for s in os.path.basename(self.filename) if s.isdigit()])
            label_name = os.path.join("Pancreas","labels","TCIA_pancreas_labels-02-05-2017","label"+file_num+".nii.gz")

            sitk_t1 = sitk.ReadImage(label_name)
            label_vol = sitk.GetArrayFromImage(sitk_t1)
        else:
            label = os.path.join("data/labels",os.path.basename(self.filename))
            
            tmp = sitk.ReadImage(label)
    
            # and access the numpy array:
            label_vol = sitk.GetArrayFromImage(tmp)
        
      
        # if not s0==s1==s2:
        #     pred = zoom(pred,self.scales)
        
        self.cur_slice = len(label_vol)-1 # pred.sum(2).sum(1).nonzero()[0][0]
        self.pred = label_vol
        
    def getPrediction(self):
        prediction = self.pred.copy()
        return prediction
        
    def undo(self):

        if len(self.oldAnn)>1 and len(self.oldResize)>1:
            self.oldAnn.pop()
            self.oldResize.pop()
            self.annotationPix = self.oldAnn[-1].copy()
            self.resizePix = self.oldResize[-1].copy()
        
        if len(self.oldAnn)>1 and len(self.oldResize)>1:
            self.predict()
        else:
            self.reset_masks()
            self.update()
            
    def reset_current_image(self):
        self.annotationPix = self.oldAnn[0]
        self.resizePix = self.oldResize[0]
        self.oldAnn,self.oldResize = [self.annotationPix.copy()], [self.resizePix.copy()]

        self.reset_masks()
        self.update()
            
    def keyPressEvent(self, event):
        if 47<event.key()<58: #numbers 0 (48) to 9 (57)
            self.label = event.key()-48
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            self.showInfo(f'Changed pen label to {self.label}')
        elif event.key()==PyQt5.QtCore.Qt.Key_Up: # uparrow          
            self.penWidth = min(self.penWidth+1,50) 
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            self.showInfo(f'Changed pen width to {self.penWidth}')
        elif event.key()==PyQt5.QtCore.Qt.Key_Down: # downarrow
            self.penWidth = max(self.penWidth-1,1)
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            self.showInfo(f'Changed pen widht to {self.penWidth}')
        elif event.key()==PyQt5.QtCore.Qt.Key_S: # s
            self.saveOutcome()
        elif event.key()==PyQt5.QtCore.Qt.Key_O: # o
            self.overlay = (self.overlay+1)%len(self.overlays)
            self.update()
            self.showInfo(f'Changed overlay to {self.overlays[self.overlay]}')
        elif event.key()==PyQt5.QtCore.Qt.Key_Z: # z
            self.undo()
            # if not self.zPressed:
            #     self.showInfo('Zooming enabled')
            #     self.zPressed = True
            #     self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
            #     self.update()
        elif event.key()==PyQt5.QtCore.Qt.Key_R: # r
            self.reset_current_image()
        elif event.key()==PyQt5.QtCore.Qt.Key_H: # h        
            if not self.hPressed:
                self.hPressed = True
                self.showHelp()
        # elif event.key()==PyQt5.QtCore.Qt.Key_Escape: # escape
        #     self.closeEvent()
        elif event.key()==PyQt5.QtCore.Qt.Key_Return:
            self.predict()
            self.showInfo('Predicting mask from skeleton')
            self.update()
        elif event.key()==PyQt5.QtCore.Qt.Key_I:
            self.reset_image()
            self.resetZoom()
            self.label = 1
            self.update()
        self.setTitle()
        
    def keyReleaseEvent(self, event):
        # if event.key()==PyQt5.QtCore.Qt.Key_Z: # z
        #     if not self.activelyZooming:
        #         self.drawCursorPoint(self.lastCursorPoint)
        #         if self.newZoomValues is None:
        #             self.resetZoom()
        #         elif self.newZoomValues==0:
        #             self.showInfo('Zooming canceled')
        #             self.newZoomValues = None
        #         else:
        #             self.executeZoom()                       
        #         self.update()
        #     self.zPressed = False
        if event.key()==PyQt5.QtCore.Qt.Key_H: # h
            self.hideText()
            self.hPressed = False
            
    # def closeEvent(self, event):
    #     self.showInfo("Bye, I'm closing")
    #     PyQt5.QtWidgets.QApplication.quit()
    #     # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
    #     # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
   
    def saveOutcome(self):
        self.annotationPix.save(self.saveAddress)
        self.showInfo(f'Saved annotations as {self.saveAddress}')
        
    # colors associated with different labels
    colors = [
        [0, 0, 0], # background, transparency is always drawn with black
        [255, 0, 0], # label 1
        [0, 191, 0], # label 2
        [0, 0, 255], # etc
        [255, 127, 0],
        [0, 255, 191],
        [127, 0, 255],
        [191, 255, 0],
        [0, 127, 255],
        [255, 64, 191]] 

    @classmethod
    def color_picker(cls, label, opacity):
        """ Pen colors given for a label number. """
        opacity_value = int(opacity*255)
        color = PyQt5.QtGui.QColor(cls.colors[label][0], cls.colors[label][1], 
                cls.colors[label][2], opacity_value)
        return(color)
    
    @staticmethod
    def formatQRect(rect):
        coords =  rect.getCoords()
        s = f'({coords[0]},{coords[1]})--({coords[2]},{coords[3]})'
        return(s)     
    
    def closeEvent(self,event):
        PyQt5.QtWidgets.QApplication.quit()
        
    def showVolumeTmp(self):
        return
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        
        vtk_data = numpy_to_vtk(num_array=self.pred.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

        imgdat = vtk.vtkImageData()
        imgdat.GetPointData().SetScalars(vtk_data)
        imgdat.SetDimensions(self.ct_shape[1], self.ct_shape[2], self.ct_shape[0])
        imgdat.SetOrigin(0, 0, 0)
        spacing = self.spacing
        imgdat.SetSpacing(spacing[0], spacing[1], spacing[2])
        
        
        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(imgdat)
        surface.ComputeNormalsOn()
        surface.SetValue(0, 0.5)
        
        renderer = vtk.vtkRenderer()
        # renderer.SetBackground(vtk.colors.GetColor3d('DarkSlateGray'))
        
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName('MarchingCubes')
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(vtk.colors.GetColor3d('MistyRose'))
        
        renderer.AddActor(actor)
        
        render_window.Render()
        interactor.Start()
 
    
def annotate(folder_name, net_name,
             resize_size = 128,
             scale_to_screen = False):
    app = PyQt5.QtWidgets.QApplication([])
    ex = Annotator.fromFolder(folder_name,
                              net_name, 
                              resize_size,
                              scale_to_screen
                              )
    # ex = Annotator.fromGrayscale(image)
    ex.show()
    app.exec()       
 
    
if __name__ == '__main__':
    
    '''
    For use from command-line. 
    '''
    

    if len(sys.argv)<2:
        print('Usage: $ python annotator.py image_filename')
    else:
        app = PyQt5.QtWidgets.QApplication([])
        filename = sys.argv[1]
        ex = Annotator.fromFilename(filename)
    
        ex.show()  # is probably better placed here than in init
        app.exec()
    
        #app.quit(), not needed? exec starts the loop which quits when the last top widget is closed  
        #sys.exit(), not needed?  
    
    