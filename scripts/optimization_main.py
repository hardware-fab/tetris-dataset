import sys
import random
sys.path.append(sys.path[0] + '/../models')
import gaussianProcess as model
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import math
import operator

#-------------------Input----------------------

dataset = sys.argv[1]    #Main dataset used for training and testing


#----------------Configurable variables--------------

#Change these three variable to experiment with different scenarios
num_test = 10                                                                                 #Number of tests performed for validation
threshold_list = {"ADPCM": 5, "AES": 11, "DFADD": 35, "DFMUL": 30, "GSM": 25, "SHA3": 350}    #Optimization thresholds for the various applications (edit them as needed)    
n_iter_max = 30                                                                               #Number of iterations performed by the optimization loop

acc_info = sys.path[0] + "/../data/accelerators.json"        #File containing accelerators' information

plot_styles = ["blue", "green"]                                          #List of colors (for the plot)
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
infinite_area = 9999999        #10 millions LUT to represent an infinite area
lut_approx = 100               #Ignore the last two digits of the LUT consumption - they are not significative


#---------------------Configuration search methods---------------------

#Identify the best configuration (the configuration with the lowest area among the ones that satisfy the thresholds)
#If no configuration satisfy the thresholds, choose one randomly
#NOTE: cannot choose a configuration that has already been tested 
def BestPredictedConfig(throughput, area, testedConfigs):
  bestArea = infinite_area
  bestConfigsSoFar = []
  for config in range(0, len(data_dict)):
    #First: check if the configuration has already been tested 
    if config in testedConfigs:
      continue
    
    #Second: verify that the configuration overcomes the threshold for all the applications
    overThreshold = True
    for app in range(0, n_app):
      overThreshold = overThreshold and (throughput[config][app] >= threshold_list[app_list[app]])
    #If it does, check its area; otherwise, use the infinite one
    if overThreshold:
      if area[config]//lut_approx < bestArea//lut_approx:
        bestConfigsSoFar = [config]
        bestArea = area[config]
      elif area[config]//lut_approx == bestArea//lut_approx:
        bestConfigsSoFar.append(config)
    else:
      if bestArea == infinite_area:
        bestConfigsSoFar.append(config)
  
  #Now, choose one configuration randomly among the ones in the list
  return bestConfigsSoFar[random.randint(0, len(bestConfigsSoFar)-1)] 

##############################################################

#Identify the best configuration among the real results
def BestRealConfig():
  bestArea = infinite_area
  bestConfigSoFar = -1
  count = 0
  for configKey, configValue in data_dict.items():
    #Check if the configuration overcomes the threshold for all the applications
    overThreshold = True
    for app in range(0, n_app):
      overThreshold = overThreshold and (configValue["throughput_total"][app_list[app]] >=  threshold_list[app_list[app]])
    
    if not overThreshold:
      count += 1
      continue

    #If it does, check its area
    if configValue["LUT"] < bestArea:
      bestConfigSoFar = count
      bestArea = configValue["LUT"]

    count += 1
  return bestConfigSoFar

##############################################################

#Return the distance from the threshold for configurations that does not satisy them
def ThresholdDistance(throughput):
  distanceFromThreshold = 0
  #For each app, detect which ones don't satisfy the threshold
  for app in range(0, n_app):
    if throughput[app] < threshold_list[app_list[app]]:
      #Add the normalized difference between the threshold and the actual value to the distance factor 
      distanceFromThreshold += (threshold_list[app_list[app]] - throughput[app])/threshold_list[app_list[app]]

  return distanceFromThreshold

##############################################################

#Find the best configuration at a given iteration
def BestConfigIter(throughputBest, areaBest, throughputNew, areaNew):
  #Check if the configuration overcomes the threshold for all the applications
  distanceFromThresholdNew = ThresholdDistance(throughputNew)
  distanceFromThresholdOld = ThresholdDistance(throughputBest)
  #If it doesn't and the best configuration already satisfies the thresholds, skip to the next
  if distanceFromThresholdNew > 0 and  distanceFromThresholdOld == 0:
    return areaBest, throughputBest
  #If it doesn't but the best configuration doesn't satisfy the threshold either, use a formula to check which one is nearer to the threshold
  elif distanceFromThresholdNew > 0:
    if distanceFromThresholdNew < distanceFromThresholdOld:
      return areaNew, throughputNew
    else:
      return areaBest, throughputBest
  #If it does overcome the threshold, compare the areas
  else:
    if areaNew < areaBest or distanceFromThresholdOld > 0:
      return areaNew, throughputNew
    else:
      return areaBest, throughputBest



#---------------------Main program---------------------

#This program simulates the execution of an iterative optimization process. At each iteration the ML model
#identifies the most promising configuration, this configuration is implemented and tested to collect real data,
#and based on the collected results the ML model predicts a new configuration for the next iteration.

#Real throughput and area results of the best onfigurations so far at each iteration
best_thr_so_far = [[[0 for x in range(0, num_test)] for x in range(0, n_iter_max)] for x in range(0, n_app)]
best_area_so_far = [[0 for x in range(0, num_test)] for x in range(0, n_iter_max)]

#Repeat for the number of tests
for test in range(0, num_test):
  print("Executing Test " + str(test))
  tested_configs = [random.randint(0, len(data_dict)-1)]                          #List of configurations already tested (initialized with one random number)
  training_key, training_value = (list(data_dict.items()))[tested_configs[0]]     #Get a dictionary from this single configuration
  training_dict = {training_key: training_value}
  model_inst = model.ModelClass(app_list, n_tiles, par_lvls, acc_dict)            #Initialize the model
  model_inst.ConfigSetInit(training_dict, data_dict)

  #Repeat for the number of iterations
  for iter in range(0, n_iter_max):
    #Train the model
    model_inst.ModelTraining()
    #Predict the upper value for the throughput and the area for each configuration
    prediction_throughput = model_inst.PredictUpper()
    prediction_area = model_inst.PredictArea()
    #Use those information to choose the next config to check, and add it to the training set
    next_config = BestPredictedConfig(prediction_throughput, prediction_area, tested_configs)
    tested_configs.append(next_config)
    training_key, training_value = (list(data_dict.items()))[next_config]
    training_dict = {training_key: training_value}  
    model_inst.AddTrainingConfig(training_dict)

    #To assess the loop performance, we save the predicted results at each iteration
    prediction_throughput = model_inst.PredictMean()
    new_predicted_config = BestPredictedConfig(prediction_throughput, prediction_area, [])
    new_config_key, new_config_value = (list(data_dict.items()))[new_predicted_config]
    new_thr = []
    for app in range(0, n_app):
      new_thr.append(new_config_value["throughput_total"][app_list[app]])
    #And then we identify the best configuration among all the results collected up to the current iteration
    #NOTE: there is no guarantee that the current prediction is the best one!
    if iter == 0:
      best_area_so_far[iter][test] = new_config_value["LUT"]
      for app in range(0, n_app):
        best_thr_so_far[app][iter][test] = new_thr[app]
    else:
      best_thr_temp = []
      for app in range(0, n_app):
        best_thr_temp.append(best_thr_so_far[app][iter-1][test])
      best_area_so_far[iter][test], best_thr_temp = BestConfigIter(best_thr_temp, best_area_so_far[iter-1][test], new_thr, new_config_value["LUT"])
      for app in range(0, n_app):
        best_thr_so_far[app][iter][test] = best_thr_temp[app]


#------------------------Average and confidence interval---------------------

#Average the throughput over the tests
thrAverage = [[0 for col in range(0, n_iter_max)] for col in range(0, n_app)]

for app in range(0, n_app):
  for iter in range(0, n_iter_max):
    #First, sum all the results  
    for test in range(0, num_test):
      thrAverage[app][iter] += best_thr_so_far[app][iter][test]
    #Second, divide for the number of tests
    thrAverage[app][iter] = thrAverage[app][iter]/num_test

#Compute the confidence interval
thrConfidenceInterval = [[0 for col in range(0, n_iter_max)] for col in range(0, n_app)]
for app in range(0, n_app):
  for iter in range(0, n_iter_max):
    #First, sum the square differences between the throughputs and the average throughput  
    for test in range(0, num_test):
      thrConfidenceInterval[app][iter] += (best_thr_so_far[app][iter][test]-thrAverage[app][iter])**2
    #Then, divide by the number of test
    thrConfidenceInterval[app][iter] = math.sqrt(thrConfidenceInterval[app][iter]/num_test)
    #Finally, compute the confidence interval by further dividing for the square root of the number of tests and multiplying by two 
    thrConfidenceInterval[app][iter] = 2*thrConfidenceInterval[app][iter]/math.sqrt(num_test)


#Average the area over the tests
areaAverage = [0 for col in range(0, n_iter_max)]

for iter in range(0, n_iter_max):
  #First, sum all the results  
  for test in range(0, num_test):
    areaAverage[iter] += best_area_so_far[iter][test]
  #Second, divide for the number of tests
  areaAverage[iter] = areaAverage[iter]/num_test

#Compute the confidence interval
areaConfidenceInterval = [0 for col in range(0, n_iter_max)]
for iter in range(0, n_iter_max):
  #First, sum the square differences between the areas and the average area  
  for test in range(0, num_test):
    areaConfidenceInterval[iter] += (best_area_so_far[iter][test]-areaAverage[iter])**2
  #Then, divide by the number of test
  areaConfidenceInterval[iter] = math.sqrt(areaConfidenceInterval[iter]/num_test)
  #Finally, compute the confidence interval by further dividing for the square root of the number of tests and multiplying by two 
  areaConfidenceInterval[iter] = 2*areaConfidenceInterval[iter]/math.sqrt(num_test)

#----------------------------Plot the results--------------------------------

#Set font and size of the letters
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': font_size})

#Main figure
fig, ax = plt.subplots(1+n_app)
x_axis = list(range(1, n_iter_max+1))

#Plot the area of the best configuration for reference
best_config = BestRealConfig()
best_config_key, best_config_value = (list(data_dict.items()))[new_predicted_config]
best_area_plot = [best_config_value["LUT"]]*n_iter_max

#Plot the threshold for reference
threshold_plot = []
for app in range(0, n_app):
  threshold_plot.append([threshold_list[app_list[app]]]*n_iter_max)


#Plot the area of the chosen configuration at each iteration
ax[0].plot(x_axis, areaAverage, color=plot_styles[0], linewidth=2)
ax[0].plot(x_axis, best_area_plot, color=plot_styles[1], linewidth=2)
#Set the legend
ax[0].legend(["Model", "Optimum"], loc="upper right")
#Plot the confidence interval as a filled, semi-transparent area
ax[0].fill_between(x_axis, list(map(operator.sub, areaAverage, areaConfidenceInterval)), list(map(operator.add, areaAverage, areaConfidenceInterval)), color=plot_styles[0], alpha=.1)
#Style the axis labels with italic mathematical expressions
ax[0].set_xlabel("n", style='italic')
ax[0].set_ylabel("$\mathdefault{\widehat{LUT}}$(n)", style='italic')

#Plot the throughput of the chosen configuration at each iteration
for app in range(0, n_app):
  ax[1+app].plot(x_axis, thrAverage[app], color=plot_styles[0], linewidth=2)
  ax[1+app].plot(x_axis, threshold_plot[app], color=plot_styles[1], linewidth=2)
  #Set the legend
  ax[1+app].legend(["Model", "Threshold"], loc="lower right")
  #Plot the confidence interval as a filled, semi-transparent area
  ax[1+app].fill_between(x_axis, list(map(operator.sub, thrAverage[app], thrConfidenceInterval[app])), list(map(operator.add, thrAverage[app], thrConfidenceInterval[app])), color=plot_styles[0], alpha=.1)
  #Style the axis labels with italic mathematical expressions
  ax[1+app].set_xlabel("n", style='italic')
  ax[1+app].set_ylabel("$\mathdefault{\widehat{thr}}$(n)", style='italic')

#Adjust the border space
plt.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.15)


plt.show()




 
 
