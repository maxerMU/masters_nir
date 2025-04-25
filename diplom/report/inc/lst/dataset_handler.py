DATASET_PATH: str = "../planes_dataset"
TRAIN_TENSORS_PATH: str = "./train_tensors"
TEST_TENSORS_PATH: str = "./test_tensors"

class DataSetHandler:
    _AUG_ROTATE_ANGLE = 30
    def TrainSize(self):
        files = listdir(TRAIN_TENSORS_PATH)
        return len(files) - 1 #file with y results

    def TestSize(self):
        files = listdir(TEST_TENSORS_PATH)
        return len(files) - 1 #file with y results

    def GetTrainingBatch(self, batchIndexes, needAug=False):
        xBatch, yBatch = self._GetBatch(batchIndexes, f"{TRAIN_TENSORS_PATH}/train")
        if needAug:
            xBatch, yBatch = self._AugmentateBatches(xBatch, yBatch)
        
        return xBatch, yBatch


    def GetTestBatch(self, batchIndexes):
        return self._GetBatch(batchIndexes, f"{TEST_TENSORS_PATH}/test")
    
    def UpdateData(self):
        self._UpdateTrainData()
        self._UpdateTestData()

    def _GetBatch(self, batchIndexes, pathPrefix):
        y = []
        with open(f"{pathPrefix}_results.txt") as classesFile:
            classes = classesFile.read().split('\n')
            for i in batchIndexes:
                y.append(int(classes[i]))
        
        x = []
        for i in batchIndexes:
            x.append(torch.load(f"{pathPrefix}_{i}.pt"))

        return torch.stack(x), torch.Tensor(y).to(torch.long)

    def _UpdateTrainData(self):
        xTrain = []
        yTrain = []
        with open(f"{DATASET_PATH}/ImageSets/Main/train.txt") as trainImagesFile:
            for imageNumber in trainImagesFile:
                images, classes = self._FindAllImages(int(imageNumber))
                xTrain = xTrain + images
                yTrain = yTrain + classes
                
        for i in range(len(xTrain)):
            tensor: torch.Tensor = self._ConverToTensor(xTrain[i])
            torch.save(tensor, f"{TRAIN_TENSORS_PATH}/train_{i}.pt")

        with open(f"{TRAIN_TENSORS_PATH}/train_results.txt", "w") as f:
            for i in range(len(yTrain)):
                f.write(f"{yTrain[i]}\n")

    def _UpdateTestData(self):
        xTest = []
        yTest = []
        with open(f"{DATASET_PATH}/ImageSets/Main/test.txt") as testImagesFile:
            for imageNumber in testImagesFile:
                images, classes = self._FindAllImages(int(imageNumber))
                xTest = xTest + images
                yTest = yTest + classes
                
        for i in range(len(xTest)):
            tensor: torch.Tensor = self._ConverToTensor(xTest[i])
            torch.save(tensor, f"{TEST_TENSORS_PATH}/test_{i}.pt")

        with open(f"{TEST_TENSORS_PATH}/test_results.txt", "w") as f:
            for i in range(len(yTest)):
                f.write(f"{yTest[i]}\n")

    def _ConverToTensor(self, imagePath: str) -> torch.Tensor:
        image: Image.Image = Image.open(imagePath)
        transform = transforms.ToTensor()
        tensor: torch.Tensor = transform(image)
        # remove alpha channel
        if (tensor.size(0) == 4):
            tensor = tensor[:-1]

        return tensor

    def _AugmentateBatches(self, xBatch: torch.Tensor, yBatch: torch.Tensor):
        xBatchAug = []
        yBatchAug = []

        for i, tensor in enumerate(xBatch):
            augTensor1 = transforms.functional.rotate(tensor, self._AUG_ROTATE_ANGLE)
            augTensor2 = transforms.functional.rotate(tensor, -self._AUG_ROTATE_ANGLE)
            augTensor3 = transforms.functional.adjust_brightness(tensor, 1.5)
            augTensor4 = transforms.functional.gaussian_blur(tensor, kernel_size=(5,9), sigma=3)
            xBatchAug += [tensor, augTensor1, augTensor2, augTensor3, augTensor4]
            yBatchAug += [yBatch[i].item()] * 5
        
        order = np.random.permutation(len(xBatchAug))
        xBatchAug = np.array(xBatchAug)[order]
        yBatchAug = np.array(yBatchAug)[order]

        return torch.stack(list(xBatchAug)), torch.Tensor(list(yBatchAug)).to(torch.long)
