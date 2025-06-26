import torch

BATCH_SIZE = 1000
BUFFER_SIZE = 128
MODEL_HIDDEN_SIZE = 512
MODEL_LSTM_HIDDEN_SIZE = 512
TRAIN_PART = 0.7
TRAIN_DATA_FOLDER = f"train_data_{BUFFER_SIZE}"
TEST_DATA_FOLDER = f"test_data_{BUFFER_SIZE}"
DATASET = f"{TRAIN_DATA_FOLDER}/tpcc_big_logfile"
DATASET_TEST_OPT = f"{TRAIN_DATA_FOLDER}/tpcc_big_logfile_test_victims"
DATASET_TRAINT_OPT = f"{TRAIN_DATA_FOLDER}/tpcc_big_logfile_train_victims"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
