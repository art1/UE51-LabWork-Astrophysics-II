########
#
# Exemple de script python qui lit le fichier de
# donnees 2016:09:21:13:07:25:21.rad et trace le spectre
#
########

import json
import numpy
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import array
import scipy.optimize as scp


# we use two different functions for our gaussian fit process
# gaussian2D is used to create a gaussian2D curve with given Parameters
# gaussian2D_CurveFit is used for the curve fitting process (we need to use ravel here)


# gives a gaussion2D for given Parameters
# see here: https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
# we use lambda function for az0 and el0
def gaussian2D(offAz, offEl, majWidth, minWidth, thetaRot, offsetZero, amp):
    # calculate the general gaussian parameters
    # to check: this could be probably done also by just calculating the mean vector
    # and 0.5/covariance_matrix with given data... ?
    a = (numpy.cos(thetaRot)**2)/(2*majWidth**2) + (numpy.sin(thetaRot)**2)/(2*minWidth**2)
    b = -(numpy.sin(2*thetaRot))/(4*majWidth**2) + (numpy.sin(2*thetaRot))/(4*minWidth**2)
    c = (numpy.sin(thetaRot)**2)/(2*majWidth**2) + (numpy.cos(thetaRot)**2)/(2*minWidth**2)
    return lambda az0,el0 : (offsetZero + amp*numpy.exp( - (a*((offAz-az0)**2) + 2*b*(offAz-az0)*(offEl-el0) + c*((offEl-el0)**2))))


# used for fitting the curve with curve_fit and leastsq - that's why we have to give the el/az data!
def gaussian2D_CurveFit(data,offAz, offEl, majWidth, minWidth, thetaRot, offsetZero, amp):
    az0,el0 = data
    a = (numpy.cos(thetaRot)**2)/(2*majWidth**2) + (numpy.sin(thetaRot)**2)/(2*minWidth**2)
    b = -(numpy.sin(2*thetaRot))/(4*majWidth**2) + (numpy.sin(2*thetaRot))/(4*minWidth**2)
    c = (numpy.sin(thetaRot)**2)/(2*majWidth**2) + (numpy.cos(thetaRot)**2)/(2*minWidth**2)
    ret = offsetZero + amp*numpy.exp( - (a*((offAz-az0)**2) + 2*b*(offAz-az0)*(offEl-el0) + c*((offEl-el0)**2)))
    return ret.ravel()


# adjusts the zerolevel to the smallest measurement, then normalizes wrt the maximum
def normalizeAndMove(z):
    res = z
    resMin = z.min()
    res -= resMin
    resMax = res.max()
    res /= resMax
    return res



filename = "2016:09:28:07:49:24:28.rad"
json_data=open(filename)
data=json.load(json_data)
series=data["series"]
json_data.close()


x = list()
y = list()
z = list()

#read the data in three arrays
i=0
azOld=-4    # -4 because that's the offset where we want to start
elOld=-4
first=True
sumVal=list()
val = list()

for spec in series:
    # ditch the first measurement, we start at -4/-4 offset (at the second measurement)
    if not(first):
        # we get the values and apply a threshold filter
        azOff = spec["azOff"]
        elOff = spec["elOff"]
        val = spec["value"]
        threshold = max(val)-(max(val)/3)
        # print "max: %d, threshold: %d" %(max(val),threshold)
        j = 0
        while j < len(val):
            if val[j] < threshold:
                del val[j]
            j += 1
        # we compare the old offset values to the current one, if they are the same
        # we just append them to the other values ( basically summing all the values with the same elevation and azimuth)
        if ((azOld==azOff) and (elOld==elOff)):
            sumVal.append(sum(val)/len(val))
        else :
        # else we append them to the x y z vectors and initialize a new list
            z.append(sum(sumVal)/len(sumVal))
            x.append(azOff)
            y.append(elOff)
            sumVal=list()
        # print "az: %d el: %d" %(azOff, elOff)
        azOld=azOff
        elOld=elOff
    else:
        first=False
    i+=1
# now we append the last summed values
z.append(sum(sumVal)/len(sumVal))
x.append(azOff)
y.append(elOff)

print x
print y
print z

###########
# pour tracer le spectre:
###########
# convert lists to numpy arrays (for the plots)
xi = numpy.asarray(x)
yi = numpy.asarray(y)
zi = numpy.asarray(z)

xi = numpy.unique(xi)
yi = numpy.unique(yi)

print len(xi)
print len(yi)
print len(zi)

# reshapes the totalflux values to match the 5x5 grid
X,Y = numpy.meshgrid(xi,yi)
Z=zi.reshape(len(yi),len(xi))

# normalizes and accounts for zero offset (so we stretch the data from 0 to 1)
Z = normalizeAndMove(Z)
data = (X,Y)

# we need az0 and el0 for later lambda parse
az0 = copy.deepcopy(X)
el0 = copy.deepcopy(Y)

# we guess initial parameters, sun should be in the middle, so azOff=elOff=0
# also I guess the zero offset should be zero...
paramGuess = [0,0,1,1,1,0,1]

# now we run curve_fit to get the predicted parameters (starting off on our guessed ones)
# curve_fit uses leastsq internally
pparam, covmat = scp.curve_fit(gaussian2D_CurveFit,data,numpy.ravel(Z),p0=paramGuess)
print "predicted params: ",pparam
print "estimated covariance matrix: ", covmat

# we create the dataset with the predicted parameters
fitted = gaussian2D(*pparam)(az0,el0)
peak = [pparam[0],pparam[1]]
print "peak: ",peak

# normalize and shift it with zeroOffset to use full range between 0 and 1
fitted = normalizeAndMove(fitted)

# plot the stuff now
plt.clf()
# here xi and yi are the vectors containing the offsets and fitted contains the fitted data
plt.pcolormesh(X, Y, fitted)
plt.colorbar()
plt.scatter(X, Y, c = fitted, s = 100, vmin = fitted.min(), vmax = fitted.max())
plt.contour(X, Y, fitted, 900)
plt.xlabel( 'azimuth offset')
plt.ylabel( 'elevation offset')
plt.title( 'Solar flow map')
# plot the peak
mark = plt.plot(peak[0],peak[1],'bx',mew=2,ms=15,label="predicted maximum")
plt.legend(numpoints=1)
plt.show()
# filenametmp=date + ".png"
# print "backup file", filenametmp
# plt.savefig (filenametmp, format = 'png')
