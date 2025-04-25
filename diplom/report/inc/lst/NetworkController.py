class NetworkController(INetworkController):
    _TRAINED_MODELS_PATH = "trained_models/"

    def __init__(self, batchSize, learningRate, needAug=True):
        self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_planesNetwork = simplenet(20).to(self.m_device)
        self.m_datasetHandler = DataSetHandler()

        self.m_batchSize = batchSize
        self.m_learningRate = learningRate
        self.m_needAug = needAug

        self.m_loss = torch.nn.CrossEntropyLoss()
        self.m_optimizer = torch.optim.Adam(self.m_planesNetwork.parameters(), lr=self.m_learningRate)
        self.m_datasetLen = self.m_datasetHandler.TrainSize()
    
    def TrainPrepare(self):
        pass

    def TrainEpoch(self):
        self.m_planesNetwork.train()
        order = np.random.permutation(self.m_datasetLen)
        for startIndex in range(0, self.m_datasetLen, self.m_batchSize):
            self.m_optimizer.zero_grad()

            xBatch, yBatch = self.m_datasetHandler.GetTrainingBatch(
                    order[startIndex:startIndex+self.m_batchSize], needAug=self.m_needAug)
            xBatch = xBatch.to(self.m_device)
            yBatch = yBatch.to(self.m_device)

            preds = self.m_planesNetwork.forward(xBatch)
            lossValue = self.m_loss(preds, yBatch)

            lossValue.backward()

            self.m_optimizer.step()
    
    def GetResults(self, xBatch):
        self.m_planesNetwork.eval()
        results = self.m_planesNetwork.forward(xBatch).argmax(dim=1)
        self.m_planesNetwork.train()

        return results
    
    def GetResult(self, imagePath):
        image: Image.Image = Image.open(imagePath)
        transform = transforms.ToTensor()
        tensor: torch.Tensor = transform(image)
        # remove alpha channel
        if (tensor.size(0) == 4):
            tensor = tensor[:-1]
        
        t = torch.stack([tensor])
        return self.GetResults(t)[0]
    
    def SaveModel(self, modelPath):
        torch.save(self.m_planesNetwork, f"{self._TRAINED_MODELS_PATH}{modelPath}")
    
    def LoadModel(self, modelPath):
        self.m_planesNetwork = torch.load(f"{self._TRAINED_MODELS_PATH}{modelPath}")
        self.m_planesNetwork.to(self.m_device)
    
    def GetAllModels(self):
        models = []
        for _, _, filenames in os.walk(self._TRAINED_MODELS_PATH):
            for model in filenames:
                models.append(model)
        
        return models
