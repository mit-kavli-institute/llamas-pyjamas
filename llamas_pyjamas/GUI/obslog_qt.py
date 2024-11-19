# Form implementation generated from reading ui file 'obslog.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_LLAMASObservingLog(object):
    def setupUi(self, LLAMASObservingLog):
        LLAMASObservingLog.setObjectName("LLAMASObservingLog")
        LLAMASObservingLog.setEnabled(True)
        LLAMASObservingLog.resize(1115, 863)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(LLAMASObservingLog.sizePolicy().hasHeightForWidth())
        LLAMASObservingLog.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(parent=LLAMASObservingLog)
        self.centralwidget.setObjectName("centralwidget")
        self.observationTable = QtWidgets.QTableWidget(parent=self.centralwidget)
        self.observationTable.setGeometry(QtCore.QRect(30, 40, 1041, 201))
        self.observationTable.setObjectName("observationTable")
        self.observationTable.setColumnCount(5)
        self.observationTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.observationTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.observationTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.observationTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.observationTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.observationTable.setHorizontalHeaderItem(4, item)
        self.observationTable.verticalHeader().setVisible(True)
        self.observationTable.verticalHeader().setDefaultSectionSize(35)
        self.refreshButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.refreshButton.setGeometry(QtCore.QRect(30, 250, 121, 32))
        self.refreshButton.setObjectName("refreshButton")
        self.headerButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.headerButton.setGeometry(QtCore.QRect(160, 250, 131, 32))
        self.headerButton.setObjectName("headerButton")
        self.datapath_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.datapath_label.setGeometry(QtCore.QRect(30, 10, 1041, 20))
        self.datapath_label.setObjectName("datapath_label")
        self.quicklookButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.quicklookButton.setGeometry(QtCore.QRect(300, 250, 171, 32))
        self.quicklookButton.setObjectName("quicklookButton")
        self.progressBar = QtWidgets.QProgressBar(parent=self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(100, 770, 981, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.plotWindow = QtWidgets.QWidget(parent=self.centralwidget)
        self.plotWindow.setGeometry(QtCore.QRect(30, 300, 1041, 451))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotWindow.sizePolicy().hasHeightForWidth())
        self.plotWindow.setSizePolicy(sizePolicy)
        self.plotWindow.setObjectName("plotWindow")
        self.plotLayout = QtWidgets.QVBoxLayout(self.plotWindow)
        self.plotLayout.setContentsMargins(0, 0, 0, 0)
        self.plotLayout.setObjectName("plotLayout")
        self.tcsOffsetButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.tcsOffsetButton.setGeometry(QtCore.QRect(480, 250, 141, 32))
        self.tcsOffsetButton.setObjectName("tcsOffsetButton")
        LLAMASObservingLog.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=LLAMASObservingLog)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1115, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(parent=self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        LLAMASObservingLog.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=LLAMASObservingLog)
        self.statusbar.setObjectName("statusbar")
        LLAMASObservingLog.setStatusBar(self.statusbar)
        self.actionData_Path = QtGui.QAction(parent=LLAMASObservingLog)
        self.actionData_Path.setObjectName("actionData_Path")
        self.menuFile.addAction(self.actionData_Path)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(LLAMASObservingLog)
        QtCore.QMetaObject.connectSlotsByName(LLAMASObservingLog)

    def retranslateUi(self, LLAMASObservingLog):
        _translate = QtCore.QCoreApplication.translate
        LLAMASObservingLog.setWindowTitle(_translate("LLAMASObservingLog", "LLAMAS Observing Log"))
        self.observationTable.setSortingEnabled(True)
        item = self.observationTable.horizontalHeaderItem(0)
        item.setText(_translate("LLAMASObservingLog", "File"))
        item = self.observationTable.horizontalHeaderItem(1)
        item.setText(_translate("LLAMASObservingLog", "Object Name"))
        item = self.observationTable.horizontalHeaderItem(2)
        item.setText(_translate("LLAMASObservingLog", "Exptime"))
        item = self.observationTable.horizontalHeaderItem(3)
        item.setText(_translate("LLAMASObservingLog", "UT Start"))
        item = self.observationTable.horizontalHeaderItem(4)
        item.setText(_translate("LLAMASObservingLog", "Airmass"))
        self.refreshButton.setText(_translate("LLAMASObservingLog", "Update List"))
        self.headerButton.setText(_translate("LLAMASObservingLog", "Display Header"))
        self.datapath_label.setText(_translate("LLAMASObservingLog", "Data Path: "))
        self.quicklookButton.setText(_translate("LLAMASObservingLog", "Display Spectral Image"))
        self.tcsOffsetButton.setText(_translate("LLAMASObservingLog", "Offset Telescope"))
        self.menuFile.setTitle(_translate("LLAMASObservingLog", "File"))
        self.menuHelp.setTitle(_translate("LLAMASObservingLog", "Help"))
        self.actionData_Path.setText(_translate("LLAMASObservingLog", "Data Path"))
