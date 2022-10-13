# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:35:25 2022

@author: lowes
"""

import sys
import vtk
import os
import numpy as np
import PyQt5
from PyQt5 import QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pydicom as dcm
from pydicom.errors import InvalidDicomError
from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from annotator import Annotator
from scipy.ndimage import gaussian_filter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, folder, dataset = False, net_name=None, resize_size=None, scale_to_screen=None, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        volShape = [960,960]
        self.folder = folder
        self.frame = QtWidgets.QFrame()
        self.hl = QtWidgets.QHBoxLayout()
        self.volFrame = Annotator.fromFolder(folder, net_name, resize_size, [960,960], dataset)

        self.surfaceFrame = QVTKRenderWindowInteractor(self.frame)
        # self.hl.addWidget(self.volFrame)
        # self.hl.addWidget(self.volFrame)
        
        self.split = QtWidgets.QSplitter(PyQt5.QtCore.Qt.Horizontal)
        self.split.addWidget(self.volFrame)
        self.split.addWidget(self.surfaceFrame)
        self.split.setSizes(volShape)
        
        self.image_viewer = None
        self.slice_text_mapper = None
        
        
        # self.volFrame.mouseRelease.connect(self.mouseReleaseEvent)
        
        
        
        self.initRender()
        # self.setup_screen_things()
        # self.setup_screen_things_3_d()
        
        
        self.hl.addWidget(self.split)
        self.frame.setLayout(self.hl)
        self.setCentralWidget(self.frame)
        self.setLayout(self.hl)
        self.showMaximized()
        self.show()
        # self.iren = self.GetInteractor()
        # self.iren.AddObserver('MouseReleaseEvent', self.mouse_release, 1.0)
        # self.iren.Initialize()
        self.iren_3d.Initialize()

    # To avoid getting OpenGL error messages when closing the program
    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        # self.volFrame.Finalize()
        self.surfaceFrame.Finalize()
        QtWidgets.QApplication.quit()

    """
    Setup a simple 3D scene with a sphere
    """
    def initRender(self):
        self.iren_3d = None
        self.actor = None
        self.ren = vtk.vtkRenderer()
        self.surfaceFrame.GetRenderWindow().AddRenderer(self.ren)
        self.iren_3d = self.surfaceFrame.GetRenderWindow().GetInteractor()
        self.renderMesh()
    def setup_screen_things_3_d(self):
        self.ren = vtk.vtkRenderer()
        self.surfaceFrame.GetRenderWindow().AddRenderer(self.ren)
        self.iren_3d = self.surfaceFrame.GetRenderWindow().GetInteractor()
        
        pred = self.volFrame.getPrediction()
        pred_filt = gaussian_filter(pred,1)
        vtk_data = numpy_to_vtk(num_array=pred_filt.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        
        self.init_pred = pred
        
        imgdat = vtk.vtkImageData()
        imgdat.GetPointData().SetScalars(vtk_data)
        imgdat.SetDimensions(self.volFrame.ct_shape[1], self.volFrame.ct_shape[2], self.volFrame.ct_shape[0])
        imgdat.SetOrigin(0, 0, 0)
        spacing = self.volFrame.spacing
        imgdat.SetSpacing(spacing[0], spacing[1], spacing[2])
        
        
        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(imgdat)
        surface.ComputeNormalsOn()
        surface.SetValue(0, 0.9 )
        surface.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
    

        self.ren.AddActor(self.actor)

    def setup_screen_things(self):
        print("Reading DICOM")
        image_data = self.read_dicom_files_carefully(self.folder)
        print("Reading done")

        if image_data is None:
            return

        self.image_viewer = vtk.vtkImageViewer2()
        self.image_viewer.SetInputData(image_data)

        self.ren_win = self.vtkWidget.GetRenderWindow()
        self.image_viewer.SetRenderWindow(self.ren_win)
        self.iren = self.image_viewer.GetRenderWindow().GetInteractor()

        style = vtk.vtkInteractorStyleImage()
        self.iren.SetInteractorStyle(style)
        self.iren.AddObserver('KeyPressEvent', self.keypress_callback, 1.0)
        self.iren.AddObserver('MouseWheelForwardEvent', self.mouse_wheel_forward_callback, 1.0)
        self.iren.AddObserver('MouseWheelBackwardEvent', self.mouse_wheel_backward_callback, 1.0)

        # Status message actor
        slice_text_prop = vtk.vtkTextProperty()
        slice_text_prop.SetFontFamilyToCourier()
        slice_text_prop.SetFontSize(20)
        slice_text_prop.SetVerticalJustificationToBottom()
        slice_text_prop.SetJustificationToLeft()

        self.slice_text_mapper = vtk.vtkTextMapper()
        msg = f"{self.image_viewer.GetSlice()} / {self.image_viewer.GetSliceMax()}"
        self.slice_text_mapper.SetInput(msg)
        self.slice_text_mapper.SetTextProperty(slice_text_prop)

        slice_text_actor = vtk.vtkActor2D()
        slice_text_actor.SetMapper(self.slice_text_mapper)
        slice_text_actor.SetPosition(15, 10)

        self.image_viewer.GetRenderer().AddActor2D(slice_text_actor)
        self.image_viewer.GetRenderer().ResetCamera()
        self.image_viewer.Render()

    def renderMesh(self):
        # ren = vtk.vtkRenderer()
        # # self.surfaceFrame = QVTKRenderWindowInteractor(self.frame)
        # self.surfaceFrame.GetRenderWindow().RemoveRenderer()
        # self.surfaceFrame.GetRenderWindow().AddRenderer(ren)
        # self.iren_3d = self.surfaceFrame.GetRenderWindow().GetInteractor()
        
        pred = self.volFrame.getPrediction()
        pred_filt = gaussian_filter(pred,1)
        vtk_data = numpy_to_vtk(num_array=pred_filt.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

        imgdat = vtk.vtkImageData()
        imgdat.GetPointData().SetScalars(vtk_data)
        imgdat.SetDimensions(self.volFrame.ct_shape[1], self.volFrame.ct_shape[2], self.volFrame.ct_shape[0])
        imgdat.SetOrigin(0, 0, 0)
        spacing = self.volFrame.spacing
        imgdat.SetSpacing(spacing[0], spacing[1], spacing[2])
        
        
        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(imgdat)
        surface.ComputeNormalsOn()
        surface.SetValue(0, 0.8 )
        surface.Update()
        
        
        surfaceTriangulator = vtk.vtkTriangleFilter()
        surfaceTriangulator.SetInputData(surface.GetOutput())
        surfaceTriangulator.PassLinesOn()
        surfaceTriangulator.PassVertsOn()
        surfaceTriangulator.Update()
        
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(surfaceTriangulator.GetOutputPort())
        smoother.SetNumberOfIterations(100)
        smoother.SetRelaxationFactor(0.5)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        
        smooth_butterfly = vtk.vtkButterflySubdivisionFilter()
        smooth_butterfly.SetNumberOfSubdivisions(3)
        smooth_butterfly.SetInputConnection(surface.GetOutputPort())
        smooth_butterfly.Update()
        
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(smooth_butterfly.GetOutputPort())
        smoother.SetNumberOfIterations(15);
        smoother.BoundarySmoothingOff();
        smoother.FeatureEdgeSmoothingOff();
        smoother.SetFeatureAngle(60.0);
        smoother.SetPassBand(0.1);
        smoother.NonManifoldSmoothingOn();
        smoother.NormalizeCoordinatesOn();
        smoother.Update()
        

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetRenderLinesAsTubes(1)
        # actor.GetProperty().SetEdgeVisibility(1);
        # actor.GetProperty().SetEdgeColor(0,0,0);

    
        self.ren.RemoveActor(self.actor)
        self.ren.AddActor(actor)
        self.iren_3d.Initialize()
        nc = vtkNamedColors()
        self.ren.SetBackground((200/255,162/255,200/255))

    def mouseReleaseEvent(self, event):
        self.volFrame.mouseReleaseEvent1(event)
        self.renderMesh()
        print("click done")
   
if __name__ == "__main__":
    app = QtWidgets.QApplication([])


    net_name = "just_learn"
    dataset = "Decathlon"    
    
    if dataset=="Pancreas":
        folder = "../data/Pancreas"
    elif dataset=="Decathlon":
        # Test folder
        folder = "../data/Decathlon/imagesTr"
    elif dataset=="Atlas":
        folder = "../data/Atlas/images"
    use_file_dialog = False
    if use_file_dialog:
        filedialog = QtWidgets.QFileDialog()
        filedialog.setNameFilter("All files (*.*)")
        filedialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        filedialog.setFileMode(QtWidgets.QFileDialog.Directory)
        # Several directories can be selected but just take the first
        if filedialog.exec():
            folder = filedialog.selectedFiles()[0]

    window = MainWindow(folder = folder,
                        dataset = dataset, 
                        net_name = net_name)
    
    window.show()
    app.exec()
