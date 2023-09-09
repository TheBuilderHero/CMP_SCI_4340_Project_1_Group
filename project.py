# cmp sci 4340
# This is just some random data for testing:

def main():
    inputDataX = [[1, -1], [1, -4], [1, -5], [1, -2]]  # if you add this pair [1, -7]
    inputDataY = [1, -1, -1, 1]  # and add this class 1 the search for a line will fail.

    # x --> note, x sub 0 starts at 1 (This is needed because when we bring in the weights we will
    # have d + 1)

    weights = [1, 1]  # w --> weights could be initialised randomly
    dimension = [0, 1]  # d --> 2d dimension

    for setLimit in range(0, 1000):  # iterates starting from 0 to 1000
        # while there is still a point that is misclassified we continue to iterate. in each iteration we take a weight
        # vector w(t) and pick a point that is currently misclassified e.g. (x(t), y(t)) and use it to update the w(t)
        # For a point to be misclassified: y(t) != sign(w^t (t)x(t)).
        #       the update rule is w(t + 1) = w(t) + y(t)x(t)
        t = 0
        # We need to get the dote product of the weight vector and the input data vector (vector of points).
        result = 0
        hasMisclassifiedPoint = False  # used for knowing when we don't update weights
        print("Data: ", inputDataX)
        print("weights: ", weights)

        for loopN in range(0, len(inputDataX)):
            result = inputDataY[loopN] * calc_dot_product(weights, inputDataX[loopN])
            print("Pass #", loopN)
            # print("info:")
            # print("X: ", inputDataX)
            print("Y= ", inputDataY[loopN])
            print("Y hat= ", calc_dot_product(weights, inputDataX[loopN]))
            print("weights before:", weights)
            # print("Result Sign: ", result)
            if result <= 0:
                print("Updating weights")
                weights = update_weight_vector(weights, inputDataX[loopN], inputDataY[loopN])
                print("weights after:", weights)
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
    else:
        print("No solution or first solution but not verified in the very last pass.")


# calculates teh dot product of two vectors:
def calc_dot_product(list1, list2):
    dot_product = 0
    for loop_n in range(0, len(list1)):
        # print("first: ", list1[loop_n], "second: ", list2[loop_n])
        # print(type(list1[loop_n]), type(list2[loop_n]))
        dot_product = dot_product + list1[loop_n] * list2[loop_n]
    print("Dot Product of ", list1, " and ", list2, " is ", dot_product)
    return dot_product


def update_weight_vector(vector_in, input_x, in_y):
    temp_list = vector_in
    stepSize = .01
    for i, w in enumerate(vector_in):
        temp_list[i] = w + stepSize * input_x[i] * in_y
    return temp_list  # returns the new weights


# call main function
main()
