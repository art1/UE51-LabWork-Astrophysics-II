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


# we use two different functions for the gaussian fit process
# gaussian2D is used to create a gaussian2D curve with given Parameters
# gaussian2D_CurveFit is only used for the curve fitting itself (we need to use ravel here)


# gives a gaussion2D for given Parameters
# see here: https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
# we use a lambda function as return and pass az0 and el0 later during the function call
# to get the gaussion fitted values for each az/el set
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

# only normalizes wrt to the maximum
def normalizeData(z):
    res = z
    resMax = res.max()
    res /= resMax
    return res

# calculate the mean for values in given array
def calculateMean(input):
    tmp = 0
    for val in input:
        tmp += val
    res = tmp /len(input)
    return res

# calculate standard deviation for the RMS
def calcSTDev(input):
    mean = calculateMean(input)
    # now calculate the standard deviation
    tmp = 0
    for val in input:
        tmp += ((val - mean) ** 2)
    tmp = tmp / len(input)
    stdev = numpy.sqrt(tmp)
    return stdev

#def getValueAt(az,el, array2D):


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
crabNebula = 968.0  # Jansky
sumVal=list()
val = list()
stdDev = list()

for spec in series:
    # ditch the first measurement, we start at -4/-4 offset (at the second measurement)
    if not(first):
        # we get the values and apply a threshold filter
        azOff = spec["azOff"]
        elOff = spec["elOff"]
        val = spec["value"]
        threshold = max(val)-(max(val)/4)
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
            # we calcluate the RMS for each measurement
            # stdDev.append(calcSTDev(val))
        else :
        # else we append them to the x y z vectors and initialize a new list
            # stdDev.append(calcSTDev(val))
            stdDev.append(calcSTDev(sumVal))
            z.append(sum(sumVal)/len(sumVal))
            x.append(azOff)
            y.append(elOff)
            sumVal = list()
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
# stdDev.append(calcSTDev(val))
print "maxVal"
maxJnsk = max(z)*570000
print maxJnsk
stdDev.append(calcSTDev(sumVal))
#print "stdDev", stdDev
#print "len stdDev", len(stdDev)




# now we average the standard deviation
standardDeviation = sum(stdDev)/len(stdDev)
print "standardDeviation", standardDeviation

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
Z = zi.reshape(len(yi),len(xi))
Z = normalizeData(Z)
data = (X,Y)

# we need az0 and el0 for the lambda function (when calculating the actual fit)
az0 = copy.deepcopy(X)
el0 = copy.deepcopy(Y)

# we guess initial parameters, sun should be in the middle, so azOff=elOff=0
# also I guess the zero offset should be zero...
paramGuess = [0,0,1,1,0,0,1]
# data,offAz, offEl, majWidth, minWidth, thetaRot, offsetZero, amp

# now we run curve_fit to get the predicted parameters (starting off on our guessed ones)
# curve_fit uses leastsq internally
# we use ravel to shape the numpy array so we can use them for the fitting function
pparam, covmat = scp.curve_fit(gaussian2D_CurveFit,data,numpy.ravel(Z),p0=paramGuess)
print "predicted params: ",pparam
print "estimated covariance matrix: ", covmat

# we create the dataset with the predicted parameters
fitted = gaussian2D(*pparam)(az0,el0)
print fitted
peak = [pparam[0],pparam[1]]
print "predicted peak: ", peak

# we normalize the fitted data again
fitted = normalizeData(fitted)

# convert to Jansky units
# Jansky at 1200 UTC at San Vito: 570 000 Jy in 1415 MHz
# we assume our maximum val equals the value measured by the San Vito station

# since we were supposed to provide a scale factor for the conversion of
# raw data with an arbitrary unit to Jansky, we calculate it in the next lines
# this is only roughly correct, since we set the maximum peak of measured data
# to be our Jansky peak - because the predicted peak in the plot is interpolated.
# we also assume that the measurements are distributed linearly
janskyScaleFactor = 570000/max(z)
print janskyScaleFactor
scaledZ = map(lambda g : g * janskyScaleFactor, z)
fitted = fitted * max(z) * janskyScaleFactor
print scaledZ
print fitted
# values seem to be correct, in range of what we expect according to
# http://www.haystack.mit.edu/edu/undergrad/srt/SRT%20Projects/

# now we're using the averaged standard deviation and calculate the RMS variations of the flux
scaledStdDev = max(stdDev) * janskyScaleFactor
print "scaledStdDev", scaledStdDev
tau = 8.0           # integration time in seconds
bandwidth = 1200000 # bandwidth in Hz
SEFD = scaledStdDev * numpy.sqrt(tau * bandwidth)
print "SEFD: ", SEFD
#print "maxFitted: ", max(fitted)
SNR = (570000 / SEFD) * numpy.sqrt(tau*bandwidth)
print "SNR", SNR

# now calculate integration time for crab nebula, to detect we need at least SNR = 1
tau_cn = (((1 * SEFD)/(crabNebula*numpy.sqrt(bandwidth)))**2)
print "crabe nebula integration time: ", tau_cn


# plot the stuff now
plt.clf()
# here xi and yi are the vectors containing the offsets and fitted contains the fitted data
plt.pcolormesh(X, Y, fitted)
bar = plt.colorbar()
bar.set_label('Jansky (10^-26 W/Hz/m^2)', rotation=270)

plt.scatter(X, Y, c = fitted, s = 100, vmin = fitted.min(), vmax = fitted.max())
plt.contour(X, Y, fitted, 900)
plt.xlabel( 'azimuth offset')
plt.ylabel( 'elevation offset')
plt.title( 'Solar flow map')
# plot the peak
mark = plt.plot(peak[0],peak[1],'bx',mew=2,ms=15,label="predicted maximum")
plt.legend(numpoints=1)
plt.show()
#filenametmp="output" + ".png"
#print "backup file", filenametmp
#plt.savefig (filenametmp, format = 'png')
