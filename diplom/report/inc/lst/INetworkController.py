class INetworkController:
    def TrainNetwork(self, epochs: int):
        self.TrainPrepare()

        trainAccs, testAccs = [], []
        for epoch in range(epochs):
            self.TrainEpoch()
            print(f"================================== {epoch} ================================")
            trainAcc, testAcc = self.LogTraining()
            trainAccs.append(trainAcc)
            testAccs.append(testAcc)
            if len(testAccs) >= 3:
                if testAccs[-1] < testAccs[-2] and testAccs[-1] < testAccs[-3]:
                    break

    def LogTraining(self):
        datasetHandler = DataSetHandler()

        predsTest = []
        yBatchTest = []
        for startTestBatch in range(0, datasetHandler.TestSize(), 500):
            xBatch, yBatch = datasetHandler.GetTestBatch(list(
                range(startTestBatch, min(startTestBatch + 500, datasetHandler.TestSize()))))
            preds = self.GetResults(xBatch)
            predsTest = predsTest + preds.tolist()
            yBatchTest = yBatchTest + yBatch.tolist()
            gc.collect()

        predsTrain = []
        yBatchTrain = []
        for startTrainBatch in range(0, datasetHandler.TrainSize(), 500):
            xBatch, yBatch = datasetHandler.GetTrainingBatch(list(range(
                startTrainBatch, min(startTrainBatch + 500, datasetHandler.TrainSize()))))
            preds = self.GetResults(xBatch)
            predsTrain = predsTrain + preds.tolist()
            yBatchTrain = yBatchTrain + yBatch.tolist()
            gc.collect()

        print("=====================================")
        testMisses = [0] * 20
        testClasses = [0] * 20
        recognizedTest = 0
        for i in range(len(predsTest)):
            testClasses[yBatchTest[i]] += 1
            if predsTest[i] == yBatchTest[i]:
                recognizedTest += 1
            else:
                testMisses[yBatchTest[i]] += 1


        trainMisses = [0] * 20
        trainClasses = [0] * 20
        recognizedTrain = 0
        for i in range(len(predsTrain)):
            trainClasses[yBatchTrain[i]] += 1
            if predsTrain[i] == yBatchTrain[i]:
                recognizedTrain += 1
            else:
                trainMisses[yBatchTrain[i]] += 1


        print("".join(map(lambda x: "{:6}".format(x), list(range(1,21)))))
        print("".join(map(lambda x: "{:6}".format(x), testClasses)))
        print("".join(map(lambda x: "{:6}".format(x), testMisses)))
        print("".join(map(lambda x: "{:6}".format(x), trainClasses)))
        print("".join(map(lambda x: "{:6}".format(x), trainMisses)))

        testAcc = float(recognizedTest) / len(predsTest)
        trainAcc = float(recognizedTrain) / len(predsTrain)
        print(float(recognizedTest) / len(predsTest))
        print(float(recognizedTrain) / len(predsTrain))
        print("=====================================")

        return trainAcc, testAcc

