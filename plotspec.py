########
#
# Exemple de script python qui lit le fichier de
# donnees 2016:09:21:13:07:25:21.rad et trace le spectre
#
########

import json
import numpy
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import array
from scipy.interpolate import interp2d



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


plt.clf()
plt.pcolormesh(X, Y, Z) # here xi and yi are the vectors containing the offsets and zi contains the stream.
#plt.show()
plt.colorbar()


plt.scatter(X, Y, c = Z, s = 100, vmin = Z.min(), vmax = Z.max())
plt.contour(X, Y, Z, 1000)
plt.xlabel( 'azimuth')
plt.ylabel( 'elevation')
plt.title( 'Solar flow map')

plt.show()
# filenametmp=date + ".png"
# print "backup file", filenametmp
# plt.savefig (filenametmp, format = 'png')
