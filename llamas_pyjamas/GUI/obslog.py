#! /Users/simcoe/.conda/envs/llamas_reduce/bin/python3

import sys, os
from pathlib import Path

# Get package root (2 levels up from GUI folder)
package_root = Path(__file__).parent.parent.parent
sys.path.append(str(package_root))


from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QWidget, QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6 import QtWidgets
from obslog_qt import Ui_LLAMASObservingLog  # type: ignore # Import the generated class
from header_qt import Ui_HeaderWidget # type: ignore
import numpy as np
from astropy.table import Table
import traceback
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import use
use("Qt5Agg")

# from ginga.qtw.ImageViewQt import CanvasView
# from ginga.util.loader import load_data

from glob import glob
from astropy.io import fits
from astropy.samp import SAMPIntegratedClient
import logging

import subprocess
import time

# In the deployExtraction method, before running extraction:
try:
    # Try to stop any running Ray instances
    subprocess.run(["ray", "stop"], check=False)
    time.sleep(1)  # Give it time to clean up
except:
    pass


class HeaderWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_HeaderWidget()
        self.ui.setupUi(self)
        self.ui.headerTable.setColumnWidth(0,150)
        self.ui.headerTable.setColumnWidth(1,300)
        self.ui.headerTable.setColumnWidth(2,400)
        self.ui.headerTable.verticalHeader().setDefaultSectionSize(12)

    def fill(self, header):
        for card in header.cards:
            self.add_header_row([card[0], card[1], card[2]])

    def add_header_row(self, row_data):
        row_position = self.ui.headerTable.rowCount()
        self.ui.headerTable.insertRow(row_position)  # Insert a new row at the end
        item = QTableWidgetItem(row_data[0])
        self.ui.headerTable.setItem(row_position, 0, item)
        try:
            item = QTableWidgetItem(row_data[1])
        except:
            item = QTableWidgetItem(' ')
        self.ui.headerTable.setItem(row_position, 1, item)
        try:
            item = QTableWidgetItem(row_data[2])
        except:
            item = QTableWidgetItem(' ')
        self.ui.headerTable.setItem(row_position, 2, item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_path     = os.environ['PWD']
        self.header_window = None

        # Set up the UI
        self.ui = Ui_LLAMASObservingLog()
        self.ui.setupUi(self)

        # Set up all connections for the event handler
        self.ui.refreshButton.clicked.connect(self.refreshObservations)
        self.ui.menuFile.triggered.connect(self.setDataPath)
        self.ui.headerButton.clicked.connect(self.showHeader)
        self.ui.quicklookButton.clicked.connect(self.showQuickLook)
        self.ui.tcsOffsetButton.clicked.connect(self.offsetTCSFromGui)
        self.ui.extractButton.clicked.connect(self.deployExtraction)

        # Formatting styles for the GUI
        self.ui.observationTable.setColumnWidth(0,300)
        self.ui.observationTable.setColumnWidth(1,200)  
        self.ui.observationTable.setColumnWidth(2,100)
        self.ui.observationTable.setColumnWidth(3,175)
        self.ui.observationTable.setColumnWidth(4,100)
        self.ui.observationTable.verticalHeader().setDefaultSectionSize(20)
        self.ui.datapath_label.setText(f'DataPath: {self.data_path}')

        self.logger = logging.getLogger()
#        self.ginga = CanvasView(logger = self.logger, render='widget')
#        self.ginga.enable_autozoom('on')
#        self.ginga.enable_autocuts('on')
#        self.ginga.ui_set_active(True)
#        self.ginga.set_bg(0.2, 0.2, 0.2)  # Dark background
#        self.ginga_widget = self.ginga.get_widget()

        self.imageviewer = 'ds9' # or ginga

        self.ds9 = SAMPIntegratedClient()
        self.ds9.connect()
        key = self.ds9.get_private_key()
        clients = self.ds9.hub.get_registered_clients(key)
        self.client_id = clients[-1]
        for c in clients:
            metadata = self.ds9.get_metadata(c)
            if (metadata['samp.name'] == 'ds9'):
                print(f"Binding client ID {c} to ds9")
                self.client_id = c

        self.configPlotWindow()
        self.plotSpectra()

    def configPlotWindow(self):
        pw = PlotWindow()
        self.canvas = FigureCanvas(Figure())
        self.ui.plotWindow.layout().addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        self.toolbar = NavigationToolbar(self.canvas, self.ui.plotWindow, coordinates=True)
        self.ui.plotWindow.layout().addWidget(self.toolbar)
        self.ax.set_xlim([pw.xmin,pw.xmax])
        self.ax.set_ylim([pw.ymin,pw.ymax])
        self.ax.set_xlabel('Wavelength (A)')
        self.ax.set_ylabel('Counts')

    def plotSpectra(self):

        # NOTE: This janky file is the same one that is distributed in the public
        # llamas-etc gitgub repo.  WE will change this to pass in extracted spectra.
        junkfile = 'SN1a_R20mag.fits'
        spectrum = Table.read(junkfile)
        wave = spectrum['wave(nm)'] * 10.0
        flux = spectrum['flux(erg/cm2/s/A)']
        self.ax.plot(wave, flux)
        self.ax.set_ylim(0,np.max(flux))

    def refreshObservations(self):
        files = glob(f'{self.data_path}/*mef.fits')
        files2 = glob(f'{self.data_path}/ifu_testpattern.fits')
        files = files+files2
        filebase = [os.path.basename(x) for x in files]
        print("Refreshing Observing Log\n")

        self.ui.observationTable.clearContents()
        self.ui.observationTable.setRowCount(0)

        for thisfile in filebase:
            hdu = fits.getheader(self.data_path+'/'+thisfile)
            try:
                exptime = hdu['exptime']
            except:
                exptime = 'Unknown'
            
            try:
                object  = hdu['OBS TARGET NAME']
            except:
                object = 'Unknown'

            try:
                airmass = hdu['TEL AIRMASS']
            except:
                airmass = 'Unknown'

            try:
                ut_start = (hdu['UTC']).split('T')[1]
            except:
                print("ERROR: bad UTC in header")
                ut_start = 'Unknown'
            print(f'{thisfile}\t{object}\t{ut_start}')
            self.add_row([thisfile,str(object),str(exptime),ut_start,'1.0'])

    def add_row(self, row_data):
        """Adds a new row with the provided data."""
        row_position = self.ui.observationTable.rowCount()
        self.ui.observationTable.insertRow(row_position)  # Insert a new row at the end

        for column, data in enumerate(row_data):
            item = QTableWidgetItem(data)
            self.ui.observationTable.setItem(row_position, column, item)

    def setDataPath(self):
        newdir = QFileDialog.getExistingDirectory(self,"Select Data Directory")
        print(f"Setting Data Path to {newdir}")
        self.data_path = newdir
        self.refreshObservations()
        self.ui.datapath_label.setText(f'DataPath: {self.data_path}')

    def showHeader(self):
        selected_row = self.ui.observationTable.currentRow()
        if (selected_row == -1):
            reply = QMessageBox.question(self, "Confirm", f'Confirm telescope move: {dx:3.1f}"E; \t {dy:3.1f}"N', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        else:
            filename = self.ui.observationTable.item(selected_row,0)
            header = fits.getheader(self.data_path+'/'+filename.text())
            if self.header_window == None:
                self.header_window = HeaderWindow()
                self.header_window.fill(header)
            self.header_window.show()
            
    def deployExtraction(self):
        
        selected_row = self.ui.observationTable.currentRow()

        # Check if row selected
        if selected_row == -1:
            QMessageBox.warning(
                self, 
                "Error", 
                "Please select a data file from the list",
                QMessageBox.StandardButton.Ok
            )
            return

        # Get filename and construct full path
        filename = self.ui.observationTable.item(selected_row, 0).text()
        filepath = os.path.join(self.data_path, filename)

        # Validate file exists
        if not os.path.exists(filepath):
            QMessageBox.warning(
                self, 
                "Error", 
                f"File not found: {filepath}",
                QMessageBox.StandardButton.Ok
            )
            return

        try:
            # Import and run extraction
            from llamas_pyjamas.GUI.guiExtract import GUI_extract, box_extract
            result = GUI_extract(filepath)

            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"Extraction completed for {filename}",
                QMessageBox.StandardButton.Ok
            )

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error",
                f"Extraction failed: {e}",
                QMessageBox.StandardButton.Ok
            )

    def showQuickLook(self):

        if (self.imageviewer == 'ds9'):

            # Check that an image file is selected and exists (or make it if not)
            selected_row = self.ui.observationTable.currentRow()
            if (selected_row == -1):
                reply = QMessageBox.question(self, "Error", f'Please select a data file from the list', QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                return()
            else:
                filename = self.ui.observationTable.item(selected_row,0).text()
                if (filename != 'ifu_testpattern.fits'):
                    reply = QMessageBox.question(self, "Error", f'Sorry, right now only displaying the test pattern', QMessageBox.StandardButton.Ok)
                    return()

            self.ds9.ecall_and_wait(self.client_id,"ds9.set","10",cmd="frame 1")
            self.ds9.ecall_and_wait(self.client_id,"ds9.set","10",cmd=f"fits {self.data_path}/ifu_testpattern.fits")
            self.ds9.ecall_and_wait(self.client_id,"ds9.set","10",cmd="zoom to fit")

            self.reg = ImageRegions()
            self.reg.drawCrosshair(self.ds9, self.client_id)
            self.reg.drawCompass(self.ds9, self.client_id)
            
        else:
            self.logger.setLevel(logging.DEBUG)
            #image = load_data(f"ifu_testpattern.fits", logger=self.logger)
            #print("Loaded data")
            #self.ginga.set_data(image)
            #print("Set data")
            #self.ginga.add_callback('cursor-changed', self.getGingaCursor)

#    def getGingaCursor(self, ginga, event, data_x, data_y):
#            print(f"{data_x} {data_y}")

    def offsetTCSFromGui(self):
        imexam_return = self.ds9.ecall_and_wait(self.client_id,"ds9.get","10",cmd="imexam")
        coords = imexam_return['samp.result']['value']
        xcursor, ycursor = coords.split(' ')
        dx = float(self.reg.x_crosshair) - float(xcursor)
        dy = float(self.reg.y_crosshair) - float(ycursor)

        reply = QMessageBox.question(self, "Confirm", f'Confirm telescope move: {dx:3.1f}"E; \t {dy:3.1f}"N', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if (reply == QMessageBox.StandardButton.Yes):
            print("Moving the telescope")
            print(f"tcscoffsetrc {dx} {dy}")
        else:
            print("Move cancelled")

    def closeEvent(self, event):
        try:
            self.ds9.disconnect()
            print("Closing sockets")
        except:
            print("All clients closed")

class PlotWindow():
    def __init__(self):
        self.xmin = 3200
        self.xmax = 10000
        self.ymin = 0
        self.ymax = 10000

class ImageRegions():

    def __init__(self):
        self.x_crosshair = 20
        self.y_crosshair = 20

    def drawCrosshair(self, sampclient, id):
        sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "line {self.x_crosshair+1} {self.y_crosshair} {self.x_crosshair+3} {self.y_crosshair} # color=red width=3"')
        sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "line {self.x_crosshair-1} {self.y_crosshair} {self.x_crosshair-3} {self.y_crosshair} # color=red width=3"')
        sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "line {self.x_crosshair} {self.y_crosshair-1} {self.x_crosshair} {self.y_crosshair-3} # color=red width=3"')
        sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "line {self.x_crosshair} {self.y_crosshair+1} {self.x_crosshair} {self.y_crosshair+3} # color=red width=3"')

    def drawCompass(self, sampclient, id):
        # sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "compass(-7,2,50) # color=red width=2"')
        rotangle = np.radians(20)
        x_anchor = -3
        y_anchor = 7
        dx  = 0
        dy  = 6
        dx_new = dx * np.cos(rotangle) - dy * np.sin(rotangle)
        dy_new = dx * np.sin(rotangle) + dy * np.cos(rotangle)
        print(dx_new, dy_new)
        sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "line {x_anchor} {y_anchor} {x_anchor+dx_new:5.3f} {y_anchor+dy_new:5.3f} # line=0 1 color=red width=3"')

        dx  = -3.5
        dy  = 0
        dx_new = dx * np.cos(rotangle) - dy * np.sin(rotangle)
        dy_new = dx * np.sin(rotangle) + dy * np.cos(rotangle)
        print(dx_new, dy_new)
        sampclient.ecall_and_wait(id,"ds9.set","10",cmd=f'region command "line {x_anchor} {y_anchor} {x_anchor+dx_new:5.3f} {y_anchor+dy_new:5.3f} # line=0 1 color=red width=3"')



if __name__ == "__main__":
  
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
