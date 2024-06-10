import sys
import random
sys.path.append(sys.path[0] + '/../models')
import json
import operator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import math

#-------------------Input----------------------

dataset = sys.argv[1]    #Main dataset used for training and testing


#----------------Configurable variables--------------

acc_info = sys.path[0] + "/../data/accelerators.json"        #File containing accelerators' information

perc_training_set = [10, 20, 30, 40, 50, 60, 70, 80, 90]     #Training percentages for cross-validation
perc_testing_set = 10                                        #Testing percentage for cross-validation 
num_test = 10                                                #Number of tests performed for cross-validation

model_list = ["GP", "RF", "k-NN"]                                        #List of model names (for the plot)
plot_styles = ["blue", "orange", "green", "darkblue", "darkred"]         #List of model colors (for the plot)
n_models = len(model_list)                                               #Number of models used
font_size =30                                                            #Size of the font (for the plot)

data_dict = {}    #Dictionary obtained by the dataset              
acc_dict = {}     #Dictionary obtained by the accelerator information file

#Generation of the two dictionaries
with open(dataset, 'r') as file:
  data_dict = json.load(file)

with open(acc_info, 'r') as file:
  acc_dict = json.load(file)


#----------------------------Auxiliary methods-----------------------------

#Read the first configuration of the dataset to get the number of tiles
def GetNumberOfTiles():
  #Get the first configuration
  config = next(iter(data_dict))
  #Tiles are separated by underscores
  return len(config.split('_'))


#Read the first configuration of the dataset to get the list of applications
def GetAppList():
  #Get the first configuration
  config = next(iter(data_dict.values()))
  #The throughput_total field contains the list of all the applications
  appList = []
  for appName, appThroughput in config["throughput_total"].items():
    appList.append(appName)
  
  return appList


#---------------------Global variables----------------------

n_tiles = GetNumberOfTiles()   #Number of free tiles on the configuration
app_list = GetAppList()        #Application set used in the dataset
par_lvls = [1, 2, 4]           #Possible parallel levels used in the dataset (it is the same for all the dataset)
n_app = len(app_list)          #Number of applications in the application set 


#---------------------Main program-----------------------------

#Results array for all the models
error_model_final =[]
confidence_model_final =[]

for model_name in model_list:
  #Choice of the model
  if model_name == "GP":
    import gaussianProcess as model
  elif model_name == "RF":
    import randomForest as model
  elif model_name =="k-NN":
    import kNearestNeighbors as model

  print("\n\nPredicting regression results with a " + model_name + " model...")

  #A 3-dimensional matrix containing all the errors: 1 dimension for the apps, 1 for the number of training set sizes, and one for the number of tests
  errorMatrix = [[[0 for depth in range (0, n_app)] for col in range(0, len(perc_training_set))] for row in range(0, num_test)]
  
  #Randomly shuffle the dictionary
  data_list = list(data_dict.items())
  random.shuffle(data_list)
  
  #In this loop, the model execute the prediction for all the possible cases and the error is collected in an array
  for test in range(0, num_test):
    size_testing_set = int(len(data_list)*perc_testing_set/100)
    #Shift-rotate the list
    data_list.extend(data_list[0:size_testing_set])
    del data_list[0:size_testing_set]
    #Take the testing set
    testing_list = data_list[len(data_list)-size_testing_set:len(data_list)]
    #Loop over the possible training sets
    for iter in range(0, len(perc_training_set)):
      #Take the training set
      size_training_set = int(len(data_list)*perc_training_set[iter]/100)
      training_list = data_list[0:size_training_set]
      #Train and execute the model
      model_inst = model.ModelClass(app_list, n_tiles, par_lvls, acc_dict)
      model_inst.ConfigSetInit(dict(training_list), dict(testing_list))
      model_inst.ModelTraining()
      errorMatrix[test][iter] = model_inst.GetPredictionError()

  
  #Here, the error is averaged over the different tests and the different applications

  #Average the tests
  errorTestAverage = [[0 for depth in range (0, n_app)] for col in range(0, len(perc_training_set))]
  #First, sum all the results
  for test in range(0, num_test):
    for iter in range(0, len(perc_training_set)):
      errorTestAverage[iter] = list(map(operator.add, errorMatrix[test][iter], errorTestAverage[iter]))
  #Second, divide for the number of tests
  for iter in range(0, len(perc_training_set)):
    for app in range(0, n_app):
      errorTestAverage[iter][app] /= num_test
  
  #Average over the apps
  errorAppAverage = [0 for col in range(0, len(perc_training_set))]
  #First, sum all the results
  for iter in range(0, len(perc_training_set)):
    for app in range(0, n_app):
      errorAppAverage[iter] += errorTestAverage[iter][app]
  #Second, divide for the number of apps
  errorAppAverage[:] = [x / n_app for x in errorAppAverage]

  
  #Here, the confidence interval is obtained for each application and then averaged

  #Compute the standard deviation
  stdDev = [[0 for depth in range (0, n_app)] for col in range(0, len(perc_training_set))]
  #Sum the square differences between the errors and the average error
  for test in range(0, num_test):
    for iter in range(0, len(perc_training_set)):
      for app in range(0, n_app):
        stdDev[iter][app] += (errorMatrix[test][iter][app]-errorTestAverage[iter][app])**2
  #Divide by the number of test
  for iter in range(0, len(perc_training_set)):
    for app in range(0, n_app):
      stdDev[iter][app] = math.sqrt(stdDev[iter][app]/num_test)

  #Compute the confidence interval by further dividing for the square root of the number of tests and multiplying by two
  confidenceInterval = [[0 for depth in range (0, n_app)] for col in range(0, len(perc_training_set))]
  for iter in range(0, len(perc_training_set)):
    for app in range(0, n_app):
      confidenceInterval[iter][app] = 2*stdDev[iter][app]/math.sqrt(num_test)

  #Average over the apps
  confidenceAppAverage = [0 for col in range(0, len(perc_training_set))]
  #Sum the results for all the apps
  for iter in range(0, len(perc_training_set)):
    for app in range(0, n_app):
      confidenceAppAverage[iter] += confidenceInterval[iter][app]
  #Divide by the number of applications
  confidenceAppAverage[:] = [x / n_app for x in confidenceAppAverage]
  
  #Finally, save the resulting error and confidence interval
  error_model_final.append(errorAppAverage)
  confidence_model_final.append(confidenceAppAverage)

  #Print the results on the console
  print("\nRegression error:")
  print(errorAppAverage)
  print("\nConfidence interval:")
  print(confidenceAppAverage)


#----------------------------Plot the results--------------------------------

#Set font and size of the letters
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': font_size})

#Main figure
fig, ax = plt.subplots()
#Plot the error
for j in range(0, n_models):
  ax.plot(perc_training_set, error_model_final[j], color=plot_styles[j], linewidth=2)
#Set the legend with the model names
ax.legend(model_list, ncol=2, loc="upper right")
#Plot the confidence interval as a filled, semi-transparent area
for j in range(0, n_models):
  ax.fill_between(perc_training_set, list(map(operator.sub, error_model_final[j], confidence_model_final[j])), list(map(operator.add, error_model_final[j], confidence_model_final[j])), color=plot_styles[j], alpha=.1)
#Style the axis labels with italic mathematical expressions
ax.set_xlabel("perc", style='italic')
ax.set_ylabel("$\mathdefault{\widehat{err}}$(perc)", style='italic')
#Adjust the border space
plt.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.15)

plt.show()

