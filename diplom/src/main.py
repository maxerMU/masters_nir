import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QFileDialog, QTableWidgetItem
)

import config
from ui.mainwindow import Ui_MainWindow
from logfile_reader import load_pages_from_json, read_optimal_results, Page
from test_model_1 import (
    get_matches,
    OptimalStrategy,
    LRUStrategy,
    ClockSweepStrategy
)


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.bufferSizeLabel.setText(f"Размер буфера - {config.BUFFER_SIZE} страниц")
        self._connect_button_signals()

        self._data = None
        self._model_results = None
        self._optimal_results = None

    def _connect_button_signals(self):
        self.loadDataButton.clicked.connect(self._on_load_data_button)
        self.loadModelButton.clicked.connect(self._on_load_model_button)
        self.loadOptimalButton.clicked.connect(self._on_load_optimal_results_button)
        self.runButton.clicked.connect(self._on_run_button)

    def _on_load_data_button(self):
        data_path = QFileDialog.getOpenFileName(
            self, "Data Path", "train_data",
            "Data file (*.json)")[0]

        if len(data_path) == 0:
            return

        self._data = load_pages_from_json(data_path)

        self._info_box(f"Загружено {len(self._data)} обращений к страницам")
        self.timestampSpinBox.setMaximum(len(self._data))

    def _on_load_model_button(self):
        if not self._check_data_loaded():
            return

        data_path = QFileDialog.getOpenFileName(
            self, "Model Path", "results",
            "Model results file (*.mdl)")[0]

        if len(data_path) == 0:
            return

        self._model_results = []
        with open(data_path) as f:
            for line in f:
                self._model_results.append(int(line))

        if len(self._data) != len(self._model_results):
            self._error_box("Модель должна соответствовать загруженной выборке")
            self._model_results = None

    def _on_load_optimal_results_button(self):
        if not self._check_data_loaded():
            return

        data_path = QFileDialog.getOpenFileName(
            self, "Optimal Results Path", "train_data",
            "Optimal results file (*.opt)")[0]

        self._optimal_results = read_optimal_results(data_path)

        if len(self._data) != len(self._optimal_results):
            self._error_box("Результаты должны соответствовать загруженной выборке")
            self._model_results = None

    def _on_run_button(self):
        if not self._check_data_loaded():
            return

        if not self._check_model_loaded():
            return

        if not self._check_optimal_loaded():
            return

        timestamp = self.timestampSpinBox.value()
        if len(self._optimal_results[timestamp]) == 0:
            next_victim = timestamp + 1
            while (
                next_victim < len(self._optimal_results) and 
                len(self._optimal_results[next_victim]) == 0
            ):
                next_victim += 1
            self._info_box(f"На заданном временом шаге нет замещений.\
                           Ближайшее замещение на временном шаге {next_victim}")
            return

        buffer = self._get_buffer(timestamp)

        optimal_res = self._get_optimal_res(buffer, timestamp)
        for i, res in enumerate(optimal_res):
            self.set_table_item(0, i+1, f"{res}")

        model_res = self._get_model_res(buffer, timestamp)
        for i, res in enumerate(model_res):
            self.set_table_item(1, i+1, f"{res}")

        lru_res = self._get_lru_res(buffer, timestamp)
        for i, res in enumerate(lru_res):
            self.set_table_item(2, i+1, f"{res}")

        clock_res = self._get_clock_res(buffer, timestamp)
        for i, res in enumerate(clock_res):
            self.set_table_item(3, i+1, f"{res}")

    def set_table_item(self, row, col, text):
        item = QTableWidgetItem(f"{text}")
        item.setTextAlignment(0x0004 | 0x0080)
        self.tableWidget.setItem(row, col, item)

    def _get_clock_res(self, buffer, timestamp):
        strategy = ClockSweepStrategy()
        get_matches(self._data[:timestamp], self._optimal_results[:timestamp], strategy)
        victim = strategy.forward(self._data[timestamp], buffer, self._optimal_results[timestamp][0][0])
        victim_page: Page = buffer[victim]

        return victim, self._get_next_acc(timestamp, victim_page)

    def _get_lru_res(self, buffer, timestamp):
        strategy = LRUStrategy()
        get_matches(self._data[:timestamp], self._optimal_results[:timestamp], strategy)
        victim = strategy.forward(self._data[timestamp], buffer, self._optimal_results[timestamp][0][0])
        victim_page: Page = buffer[victim]

        return victim, self._get_next_acc(timestamp, victim_page)

    def _get_model_res(self, buffer, timestamp):
        victim = self._model_results[timestamp]
        victim_page: Page = buffer[victim]

        return victim, self._get_next_acc(timestamp, victim_page)

    def _get_optimal_res(self, buffer, timestamp):
        victim = self._optimal_results[timestamp][0][0]
        victim_page: Page = buffer[victim]

        return victim, self._get_next_acc(timestamp, victim_page)

    def _get_next_acc(self, timestamp: int, victim_page: Page):
        i = timestamp + 1
        while (
            i < len(self._data) and
            self._data[i].get_page_id() != victim_page.get_page_id()
        ):
            i += 1

        return i - timestamp

    def _get_buffer(self, timestamp: int):
        optimal_strart = OptimalStrategy(self._optimal_results)
        _, buffer = get_matches(
            self._data[:timestamp],
            self._optimal_results[:timestamp],
            optimal_strart
        )

        return buffer

    def _check_data_loaded(self):
        loaded = self._data is not None and len(self._data) > 0
        if not loaded:
            self._error_box("Необходимо предварительно загрузить выборку")

        return loaded

    def _check_model_loaded(self):
        loaded = self._model_results is not None and len(self._model_results) > 0
        if not loaded:
            self._error_box("Необходимо предварительно загрузить модель")

        return loaded

    def _check_optimal_loaded(self):
        loaded = self._optimal_results is not None and len(self._optimal_results) > 0
        if not loaded:
            self._error_box("Необходимо предварительно загрузить оптимальный результаты")

        return loaded

    def _error_box(self, text):
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setText(text)
        box.setWindowTitle("Ошибка")
        box.exec_()

    def _info_box(self, text):
        box = QMessageBox()
        box.setIcon(QMessageBox.Information)
        box.setText(text)
        box.setWindowTitle("Информация")
        box.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
