# cmp sci 4340
# This is just some random data for testing:
#This is the working one

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#creates linearly sepearable data
def createLinear():
    numSamples = 25

    plusX = np.round(np.random.uniform(0, 5, numSamples), 2)
    plusY = np.round(np.random.uniform(0, 5, numSamples), 2)

    negX = np.round(np.random.uniform(7, 10, numSamples), 2)
    negY = np.round(np.random.uniform(7, 10, numSamples), 2)

    plusClass = np.ones(numSamples)
    

    classPos = np.column_stack((plusClass,plusX, plusY))
    classNeg = np.column_stack((plusClass,negX, negY))
    dataPoints = np.vstack((classPos, classNeg))
    yClass = np.hstack((np.ones(numSamples), -np.ones(numSamples)))
    #print(dataPoints[0:5])
    #print(yClass)

    colors = ['blue' if i == 1 else 'orange' for i in yClass]
    legend_labels = [
        mpatches.Patch(color='blue', label='Blue = \'+\''),
        mpatches.Patch(color='orange', label='Orange = \'-\'')
    ]
    plt.scatter(dataPoints[:,1],dataPoints[:,2], c=colors)
    plt.title("Linearly separable Data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(handles=legend_labels)
    plt.show()
    return dataPoints,yClass
#creates randomized non linear data
def notLinear():
    numSamples = 25
    plusX = np.round(np.random.uniform(0, 10, numSamples), 2)
    plusY = np.round(np.random.uniform(0, 10, numSamples), 2)

    negX = np.round(np.random.uniform(0, 10, numSamples), 2)
    negY = np.round(np.random.uniform(0, 10, numSamples), 2)
    holderClass = np.ones(numSamples)
     #assign a random value
    yClass1 = np.random.choice([-1, 1], size=numSamples)
    yClass2= np.random.choice([-1, 1], size=numSamples)
    classPos = np.column_stack((holderClass,plusX, plusY))
    classNeg = np.column_stack((holderClass,negX, negY))
    dataPointsNonL = np.vstack((classPos, classNeg))
    #print(dataPointsNonL)
    yClass = np.concatenate((yClass1, yClass2))
    

    #print("First five lines of dataPointsNonLinear\n", dataPointsNonL[0:5])

    colors1 = ['blue' if i == 1 else 'orange' for i in yClass1]
    colors2 = ['blue' if i == 1 else 'orange' for i in yClass2]


    legend_labels = [
        mpatches.Patch(color='blue', label='Blue = \'+\''),
        mpatches.Patch(color='orange', label='Orange = \'-\'')
    ]
   
   # Scatter plot for classPos (blue points)
    plt.scatter(classPos[:, 1], classPos[:, 2], c=colors1, label='Class Pos')

      # Scatter plot for classNeg (orange points)
    plt.scatter(classNeg[:, 1], classNeg[:, 2], c=colors2, label='Class Neg')

    

    plt.title("Non Linearly separable Data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(handles=legend_labels)

    plt.show()
    return dataPointsNonL,yClass

#plots the decision line
def plotDecsionBoundry(weights, dataPoints, yClass):
    # Calculate the slope (m) and y-intercept (b) for the line
    m = -weights[1] / weights[2]
    b = -weights[0] / weights[2]

    # Generate x values for the decision boundary
    xValues = np.linspace(0, 10, 100)  # Adjust the range as needed
    
    # Print the equation of the decision boundary
    print(f"The fitting decision line's formula is: y = {m:.2f}x + {b:.2f}")

    # Calculate corresponding y values for the decision boundary
    formula = m * xValues + b
    legendLabels = [
        mpatches.Patch(color='blue', label='Blue = \'+\''),
        mpatches.Patch(color='orange', label='Orange = \'-\'')
    ]
    # Plot the data points
    colors = ['blue' if i == 1 else 'orange' for i in yClass]
    plt.scatter(dataPoints[:, 1], dataPoints[:, 2], c=colors)

    # Plot the decision boundary
    plt.plot(xValues, formula, 'r--', label='Decision Boundary')

    plt.title("Linearly separable Test Data with Decision Boundary")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(handles=legendLabels)
    plt.show()

def main(data, yClass):
    inputDataX = data
    #print(inputDataX)
    inputDataY= yClass


    # x --> note, x sub 0 starts at 1 (This is needed because when we bring in the weights we will
    # have d + 1)

    weights = [2, 2, 2]  # w --> weights could be initialised randomly
    dimension = 2  # d --> 2d dimension

    for setLimit in range(0, 1000):  # iterates starting from 0 to 1000
        # while there is still a point that is misclassified we continue to iterate. in each iteration we take a weight
        # vector w(t) and pick a point that is currently misclassified e.g. (x(t), y(t)) and use it to update the w(t)
        # For a point to be misclassified: y(t) != sign(w^t (t)x(t)).
        #       the update rule is w(t + 1) = w(t) + y(t)x(t)
        t = 0
        # We need to get the dote product of the weight vector and the input data vector (vector of points).
        result = 0
        hasMisclassifiedPoint = False  # used for knowing when we don't update weights

        for loopN in range(0, len(inputDataX)):
            #print(inputDataX[loopN])
            result = inputDataY[loopN] * calc_dot_product(weights, inputDataX[loopN])
            #print("Pass #", loopN, "of loop #", setLimit)
            # print("info:")
            # print("X: ", inputDataX)
            #print("Y= ", inputDataY[loopN])
            #print("Y hat= ", calc_dot_product(weights, inputDataX[loopN]))
            #print("weights before:", weights)
            # print("Result Sign: ", result)
            if result <= 0:
                #print("Updating weights")
                weights = update_weight_vector(weights, inputDataX[loopN], inputDataY[loopN])
                #print("weights after:", weights)
                hasMisclassifiedPoint = True
        # This calculates the dot product of the weights and input data
        # if result >= 0:
        # not misclassified
        # else:
        # misclassified

        # ignore this for right now:

        if not hasMisclassifiedPoint:  # just so the loop ends
            print("FOUND LINE!")
            break  # because we found a line that works for the data.

        # First initalise the weights (it is always going to be w0*x0 ... wn*xn this is the equation of a line)
        # just add up the new dimensions and then add in the weights
        # stay focused on the 2d
        # always begin by initallizgng the weights. all 0s or something else
        # note d + 1 for the number of w's
        # This gives us a straight line in our hands.
        # Now check if it is on the corret side of the line
        # if it is not then we change the weight by an algebraic quantity.
        # If line 5 is less than 0 then we know that the signs are not equivilent.
        # if yj and yj hat are opposite signs then we need to update the weight
        # y hat is the sign of vector w and x vector
        # if the signs are not equivilant of yj and yj hat are not equivilent then we need to move the line.
        # alter slop intercept by modifiying the w's

        # Perceptrons are kinda confusing in textbooks. PLA is usually more than one data input. But in this case we are
        # only using one.
    print("The weights of a found line or close to the line are: ", weights)
    if not hasMisclassifiedPoint:
        print("There is a solution!")
        plotDecsionBoundry(weights,data,inputDataY)
    else:
        print("No solution or first solution but not verified in the very last pass.")


# calculates teh dot product of two vectors:
def calc_dot_product(list1, list2):
    dot_product = 0
    for loop_n in range(0, len(list1)):
        # print("first: ", list1[loop_n], "second: ", list2[loop_n])
        # print(type(list1[loop_n]), type(list2[loop_n]))
        dot_product = dot_product + list1[loop_n] * list2[loop_n]
    #print("Dot Product of ", list1, " and ", list2, " is ", dot_product)
    return dot_product


def update_weight_vector(vector_in, input_x, in_y):
    temp_list = vector_in
    stepSize = .02  # maybe need to adjust this.
    for i, w in enumerate(vector_in):
        temp_list[i] = w + stepSize * input_x[i] * in_y
    return temp_list  # returns the new weights



if __name__ == "__main__":
    #Test with linearly seperable data, after it finds a line, we will test it on a larger data set
    data, yClass = createLinear()
    #print(data)
    main(data,yClass)
    data,yClass = notLinear()
    main(data,yClass)
    
