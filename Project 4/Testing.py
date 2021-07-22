from NeuralNetUtil import buildExamplesFromCarData, buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt


def average(argList):
    return sum(argList) / float(len(argList))


def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val - mean), 2) for val in argList]
    return sqrt(sum(diffSq) / len(argList))


penData = buildExamplesFromPenData()


def testPenData(hiddenLayers=[24]):
    return buildNeuralNet(penData, maxItr=200, hiddenLayerList=hiddenLayers)


carData = buildExamplesFromCarData()


def testCarData(hiddenLayers=[16]):
    return buildNeuralNet(carData, maxItr=200, hiddenLayerList=hiddenLayers)



# question 5
penResultsq5 = []
carResultsq5 = []
# generate data for both pen and car 5 times
for num in range(5):
    nnet, penNums = testPenData()
    nnet, carNums = testCarData()
    penResultsq5.append(penNums)
    carResultsq5.append(carNums)

# run question 6
penResultsq6 = {}
carResultsq6 = {}
perceptrons = 0
while perceptrons <= 40:
    penSubResults = []
    carSubResults = []
    # generate data for both pen and car 5 times
    for num in range(5):
        nnet, penNums = testPenData([perceptrons])
        nnet, carNums = testCarData([perceptrons])
        penSubResults.append(penNums)
        carSubResults.append(carNums)
    penResultsq6[perceptrons] = penSubResults
    carResultsq6[perceptrons] = carSubResults
    perceptrons += 5

# Give Data
print("QUESTION 5")
print("Pen Results: Max - %f  Average - %f  Standard Deviation - %f" % (max(penResultsq5), average(penResultsq5),
                                                                        stDeviation(penResultsq5)))
print("Car Results: Max - %f  Average - %f  Standard Deviation - %f" % (max(carResultsq5), average(carResultsq5),
                                                                        stDeviation(carResultsq5)))

print("\nQUESTION 6")
print("Pen Data:")
for num in range(0, 41, 5):
    print("%d Perceptrons: Max - %f  Average - %f  Standard Deviation - %f" % (num, max(penResultsq6[num]),
                                                                               average(penResultsq6[num]),
                                                                               stDeviation(penResultsq6[num])))
print("Car Data:")
for num in range(0, 41, 5):
    print("%d Perceptrons: Max - %f  Average - %f  Standard Deviation - %f" % (num, max(carResultsq6[num]),
                                                                               average(carResultsq6[num]),
                                                                               stDeviation(carResultsq6[num])))