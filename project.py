# cmp sci 4340
inputData = [1, 2, 3, 4,
             5]  # x --> note, x sub 0 starts at 1 (This is needed because when we bring in the weights we will have d + 1)
weights = [1, 1, 1, 1, 1]  # w --> weights could be initialised randomly
dimension = [0, 1, 2]  # d --> 2d dimension
hasMisclassifiedPoint = True

while hasMisclassifiedPoint:  # iterates starting from 0 to some number
    # while there is still a point that is misclassified we continue to iterate. in each iteration we take a weight
    # vector w(t) and pick a point that is currently misclassified e.g. (x(t), y(t)) and use it to update the w(t)
    # For a point to be misclassified: y(t) != sign(w^t (t)x(t)).
    #       the update rule is w(t + 1) = w(t) + y(t)x(t)
    t = 0
    # We need to get the dote product of the weight vector and the input data vector (vector of points).
    loopN = 0
    result = 0

    print("Data: ", inputData)
    print("weights: ", weights)

    for data in inputData:
        result += weights[loopN] * inputData[loopN]

        loopN += 1  # increment by 1
    # This calculates the dot product of the weights and input data
    print("Dot Product", result)
    # if result >= 0:
    # not misclassified
    # else:
    # misclassified
    hasMisclassifiedPoint = False  # just so the loop ends

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

