#! /Users/simcoe/.conda/envs/llamas/bin/python3

import sys, os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QWidget
from obslog_qt import Ui_LLAMASObservingLog  # Import the generated class
from header_qt import Ui_HeaderWidget

from glob import glob
from astropy.io import fits
from astropy.samp import SAMPIntegratedClient, SAMPHubServer

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

        # Formatting styles for the GUI
        self.ui.observationTable.setColumnWidth(0,300)
        self.ui.observationTable.setColumnWidth(1,200)  
        self.ui.observationTable.setColumnWidth(2,100)
        self.ui.observationTable.setColumnWidth(3,175)
        self.ui.observationTable.setColumnWidth(4,100)
        self.ui.observationTable.verticalHeader().setDefaultSectionSize(20)
        self.ui.datapath_label.setText(f'DataPath: {self.data_path}')

        # self.samp_hub = SAMPHubServer()
        # self.samp_hub.start()

    def refreshObservations(self):
        files = glob(f'{self.data_path}/*mef.fits')
        filebase = [os.path.basename(x) for x in files]
        print("Refreshing Observing Log\n")
        for thisfile in filebase:
            hdu = fits.getheader(self.data_path+'/'+thisfile)
            exptime = hdu['exptime']
            object  = hdu['OBS TARGET NAME']
            airmass = hdu['TEL AIRMASS']
            try:
                ut_start = (hdu['UTC']).split('T')[1]
            except:
                print("ERROR: bad UTC in header")
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
            print("No row selected!")
        else:
            filename = self.ui.observationTable.item(selected_row,0)
            header = fits.getheader(self.data_path+'/'+filename.text())
            if self.header_window == None:
                self.header_window = HeaderWindow()
                self.header_window.fill(header)
            self.header_window.show()

    def showQuickLook(self):
        print("Making quick look")
        params = {}
        params['url'] = f"file://{self.data_path}/IFU_testpattern.fits"
        params['name'] = "Test image"
        message = {}
        message['samp.mtype'] = 'image.load.fits'
        message['samp.params'] = params
        client = SAMPIntegratedClient()
        client.connect()
        client.notify_all(message)
        client.disconnect()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
