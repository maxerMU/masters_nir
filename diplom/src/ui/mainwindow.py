# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(962, 464)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 5, 0, 1, 1)
        self.runButton = QtWidgets.QPushButton(self.groupBox)
        self.runButton.setObjectName("runButton")
        self.gridLayout_2.addWidget(self.runButton, 6, 0, 1, 2)
        self.timestampSpinBox = QtWidgets.QSpinBox(self.groupBox)
        self.timestampSpinBox.setObjectName("timestampSpinBox")
        self.gridLayout_2.addWidget(self.timestampSpinBox, 5, 1, 1, 1)
        self.loadModelButton = QtWidgets.QPushButton(self.groupBox)
        self.loadModelButton.setObjectName("loadModelButton")
        self.gridLayout_2.addWidget(self.loadModelButton, 2, 0, 1, 2)
        self.loadDataButton = QtWidgets.QPushButton(self.groupBox)
        self.loadDataButton.setObjectName("loadDataButton")
        self.gridLayout_2.addWidget(self.loadDataButton, 1, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 8, 1, 1, 1)
        self.loadOptimalButton = QtWidgets.QPushButton(self.groupBox)
        self.loadOptimalButton.setObjectName("loadOptimalButton")
        self.gridLayout_2.addWidget(self.loadOptimalButton, 3, 0, 1, 2)
        self.bufferSizeLabel = QtWidgets.QLabel(self.groupBox)
        self.bufferSizeLabel.setText("")
        self.bufferSizeLabel.setObjectName("bufferSizeLabel")
        self.gridLayout_2.addWidget(self.bufferSizeLabel, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox_2)
        self.tableWidget.setRowCount(4)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(3, 0, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setHighlightSections(True)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(80)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.gridLayout.addWidget(self.groupBox_2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Метод замещения страниц"))
        self.groupBox.setTitle(_translate("MainWindow", "Ввод данных"))
        self.label.setText(_translate("MainWindow", "Номер обращения"))
        self.runButton.setText(_translate("MainWindow", "Выбрать страницы для замещений"))
        self.loadModelButton.setText(_translate("MainWindow", "Загрузить модель"))
        self.loadDataButton.setText(_translate("MainWindow", "Загрузить выборку"))
        self.loadOptimalButton.setText(_translate("MainWindow", "Загрузить оптимальные результаты"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Результаты"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Метод"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Выбранная страница"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Ближайшее обращение"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Оптимальный"))
        item = self.tableWidget.item(1, 0)
        item.setText(_translate("MainWindow", "Обученная модель"))
        item = self.tableWidget.item(2, 0)
        item.setText(_translate("MainWindow", "LRU"))
        item = self.tableWidget.item(3, 0)
        item.setText(_translate("MainWindow", "Clock"))
        self.tableWidget.setSortingEnabled(__sortingEnabled)
