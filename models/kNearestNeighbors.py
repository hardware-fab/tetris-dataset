from sklearn.neighbors import KNeighborsRegressor
import numpy as np


#Tile class, characterized by a type, a set of parameters (parallelism, occupation, and memory availability) and the tile throughput
class TileClass():
  def __init__(self):
    self.type = ""
    self.param = []
    self.thr = 0

#Config class, characterized by an area metric (the number of LUTs), a total throughput, and a list of tiles
class ConfigClass():
  def __init__(self, nApp):
    self.LUT = 0
    self.total_perf = [0]*nApp
    self.tile = []



#----------------------------MODEL-----------------------------

#Model class, containing various dataset information, the training and the testing datasets, and the trained model
class ModelClass():
  def __init__(self, appList, nTiles, parLvls, accDict):
    self.appList = appList
    self.nTiles = nTiles
    self.nApp = len(appList)
    self.parLvls = parLvls
    self.accDict = accDict
    self.trainingConfig = []
    self.testingConfig = []
    self.predictorModel = []

  
  #--------------------------Auxiliary methods--------------------------

  #A method that converts an entry of the main dictionary to a ConfigClass
  def DictEntry2ConfigClass(self, dictKey, dictValue):
    
    config = ConfigClass(self.nApp)
    
    #List of the tiles in the configuration
    configTiles = dictKey.split("_")

    #Get the number of memories
    nMem = 0; #number of memories in a configuration
    for tile in range(0, len(configTiles)):
      if configTiles[tile] == 'MEM':
        nMem = nMem + 1
      config.tile.append(TileClass())

    #Find the compute-bound and memory-bound occupation, for each memory
    occMemoryBound = [0]*nMem
    occComputeBound = [0]*nMem
    accCount = 0
    for tile in range(0, len(configTiles)):
      for parLvl in self.parLvls:
        if ("x" + str(parLvl)) in configTiles[tile]:
          #NOTE: accelerators are assigned to single memory in a round-robin fashion
          if (self.accDict[configTiles[tile].replace("x" + str(parLvl), "")]["memory_bound"] == 1):
            occMemoryBound[accCount%nMem] = occMemoryBound[accCount%nMem] + 1
          else:
            occComputeBound[accCount%nMem] = occComputeBound[accCount%nMem] + 1
          accCount += 1

    #Area and total throughput
    config.LUT = int(dictValue["LUT"])
    for app in range(0, self.nApp):
      if self.appList[app] in dictValue["throughput_total"]:
        config.total_perf[app] = dictValue["throughput_total"][self.appList[app]]

    #Define the set of parameters for each tile (mem and empty tiles have no parameters)
    appCount = [0]*self.nApp
    accCount = 0
    for tile in range(0, len(configTiles)):
      #MEM tile
      if configTiles[tile] == 'MEM':
        config.tile[tile].type = 'mem'
        config.tile[tile].thr = 0
      #EMPTY tile
      elif configTiles[tile] == 'EMPTY':
        config.tile[tile].type = 'emp'
        config.tile[tile].thr = 0
      #ACC tile
      else:
        for app in range(0, self.nApp):
          #Assign type and throughput
          if self.appList[app] in configTiles[tile]:
            config.tile[tile].type = self.appList[app]
            config.tile[tile].thr = dictValue["throughput"][self.appList[app] + "_" + str(appCount[app])]
            appCount[app] = appCount[app] + 1
        #Assign the parameters
        for parLvl in self.parLvls:
          if ("x" + str(parLvl)) in configTiles[tile]:
            config.tile[tile].param = [parLvl, occComputeBound[accCount%nMem], occMemoryBound[accCount%nMem]]

        accCount += 1

    return config

  #################################

  #A function to convert a dataset from a raw matrix to a Config class list
  def ConfigGeneration(self, dataDict):
    config = [ConfigClass(self.nApp) for i in range(0, len(dataDict))]
    configCount = 0
    for key, value in dataDict.items():
      config[configCount] = self.DictEntry2ConfigClass(key, value)
      configCount += 1

    return config
  
  #################################

  #Initialization of the training and testing sets
  def ConfigSetInit(self, trainingDict, testingDict):
    self.trainingConfig = self.ConfigGeneration(trainingDict)
    self.testingConfig = self.ConfigGeneration(testingDict)

  #################################

  #Extend the training dictionary with another part of the dataset
  def AddTrainingConfig(self, trainingDict):
    newTrainingConfig = self.ConfigGeneration(trainingDict)
    self.trainingConfig.extend(newTrainingConfig)
  
  #################################

  #Training of the model, using the training set
  def ModelTraining(self):
    #These are the inputs of the gaussian predictor
    X = []
    y = []
    
    for app in range(0, self.nApp):
      X.append([])
      y.append([])
    #Fill the X and y arrays with all the available results
    for config in range(0, len(self.trainingConfig)):
      for tile in range(0, len(self.trainingConfig[config].tile)):
        for app in range(0, self.nApp):
          if self.trainingConfig[config].tile[tile].type == self.appList[app]:
            X[app].append(self.trainingConfig[config].tile[tile].param)
            y[app].append(self.trainingConfig[config].tile[tile].thr)

    #Train a model for each application using the fit method
    self.predictorModel = []
    for app in range(0, self.nApp):
      self.predictorModel.append(KNeighborsRegressor().fit(X[app], y[app]))

  #################################

  #A method to return the prediction for all the testing configurations  
  def PredictMean(self):
    #Main prediction variable
    thr_mean = [ [0 for col in range(0, self.nApp)] for row in range(0, len(self.testingConfig))]
    #Iterate over all the tiles of all the configurations
    for config in range(0, len(self.testingConfig)):
      for tile in self.testingConfig[config].tile:
        if tile.type != "mem" and tile.type != "emp":
          for app in range(0, self.nApp):
            if tile.type == self.appList[app]:
              #Predict the throughput of each tile
              inputConfig = np.array(tile.param)
              res = self.predictorModel[app].predict(inputConfig.reshape(1, -1))
              mu = np.mean(res)
              #Sum the throughput of tiles that implement the same application (in the same configuration)
              thr_mean[config][app] += mu

    return thr_mean

  #################################

  #A method to return the most optimistic prediction for all the testing configurations
  def PredictUpper(self):
    #Main prediction variable
    thr_upper = [ [0 for col in range(0, self.nApp)] for row in range(0, len(self.testingConfig))]
    #Iterate over all the tiles of all the configurations
    for config in range(0, len(self.testingConfig)):
      for tile in self.testingConfig[config].tile:
        if tile.type != "mem" and tile.type != "emp":
          for app in range(0, self.nApp):
            if tile.type == self.appList[app]:
              #Predict the throughput of each tile
              inputConfig = np.array(tile.param)
              res = self.predictorModel[app].predict(inputConfig.reshape(1, -1))
              mu = np.mean(res)
              sigma = np.std(res)
              #Sum the throughput of tiles that implement the same application (in the same configuration), plus the 5 x sigma uncertainty value
              thr_upper[config][app] += mu + 5*sigma
    return thr_upper

  #################################

  #A method to predict the area
  #DISCLAIMER: no learning methodology here: it is just a plain addition of the LUT consumption of all the tiles taken in isolation.
  #Also, it returns just the relative LUT consumption, taking into account only the tiles that changes between the various configurations.
  def PredictArea(self):
    area = [0 for i in range(0, len(self.testingConfig))]
    #Iterate over each configuration of the testing config
    for config in range(0, len(self.testingConfig)):
      for tile in self.testingConfig[config].tile:
        #Detect the type of tile and add the area indicated on the accelerator information file
        if tile.type == "mem": 
          area[config] += self.accDict["MEM"]["LUT"]
        elif tile.type == "emp": 
          area[config] += self.accDict["EMPTY"]["LUT"]
        else:
          area[config] += self.accDict[tile.type][str(tile.param[0]) + "X"]["LUT"]
    return area

  #################################

  #Method to compute the average prediction error for each app, over the whole testing set
  def GetPredictionError(self): 
    averageError = [0]*self.nApp
    #Predict the throughput for all the testing configurations
    thr_mean = self.PredictMean()
    #Compare the predicted throughput with the real one, and get the absolute error
    n_config = 0
    for config in self.testingConfig:
      for app in range(0, self.nApp):
        averageError[app] +=  abs((thr_mean[n_config][app]-config.total_perf[app])/config.total_perf[app])
      n_config = n_config + 1
    #Divide by the number of configurations to get the average
    averageError[:] = [x / n_config for x in averageError]
    
    return averageError

  #################################

  #Debugging function that prints the testing configurations in a human-readable format
  def PrintOutTrainingConfigs(self):
    for config in range(0, len(self.trainingConfig)):
      print("Config " + str(config) + ":     area = " + str(self.trainingConfig[config].LUT) + "    total_perf = " + str(self.trainingConfig[config].total_perf))
      for tile in range(0, len(self.trainingConfig[config].tile)):
        if self.trainingConfig[config].tile[tile].type =="mem" or self.trainingConfig[config].tile[tile].type =="emp":
          print("    - Tile " + str(tile) + ":    type = " + self.trainingConfig[config].tile[tile].type + "    thr = " + str(self.trainingConfig[config].tile[tile].thr))
        else:
          print("    - Tile " + str(tile) + ":    type = " + self.trainingConfig[config].tile[tile].type + "    par = " + str(self.trainingConfig[config].tile[tile].param[0]) + 
                "    occ_computeBound = " + str(self.trainingConfig[config].tile[tile].param[1]) + "    occ_memoryBound = " + str(self.trainingConfig[config].tile[tile].param[2]) + 
                "    thr = " + str(self.trainingConfig[config].tile[tile].thr))

  #################################

  #Debugging function that prints the training configurations in a human-readable format
  def PrintOutTrainingConfigs(self):
    for config in range(0, len(self.trainingConfig)):
      print("Config " + str(config) + ":     area = " + str(self.trainingConfig[config].LUT) + "    total_perf = " + str(self.trainingConfig[config].total_perf))
      for tile in range(0, len(self.trainingConfig[config].tile)):
        if self.trainingConfig[config].tile[tile].type =="mem" or self.trainingConfig[config].tile[tile].type =="emp":
          print("    - Tile " + str(tile) + ":    type = " + self.trainingConfig[config].tile[tile].type + "    thr = " + str(self.trainingConfig[config].tile[tile].thr))
        else:
          print("    - Tile " + str(tile) + ":    type = " + self.trainingConfig[config].tile[tile].type + "    par = " + str(self.trainingConfig[config].tile[tile].param[0]) + 
                "    occ_computeBound = " + str(self.trainingConfig[config].tile[tile].param[1]) + "    occ_memoryBound = " + str(self.trainingConfig[config].tile[tile].param[2]) + 
                "    thr = " + str(self.trainingConfig[config].tile[tile].thr)) 
