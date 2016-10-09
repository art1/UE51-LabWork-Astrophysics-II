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
totflux=0.0
ns = 0

x = list()
y = list()
z = list()

#read the data in three arrays
i=0
azOld=0
elOld=0
for spec in series:
    # date = spec["date"]
    # date = (date.rsplit(':',1))[0]
    # mode = spec["digital"]
    azOff = spec["azOff"]
    elOff = spec["elOff"]

    # use only every third value
    if (azOff != azOld):
        print "\n"
        # print "appending "
        # print i
        # get maximum value in one value set, and ditch the outer (small) values
        val = spec["value"]
        threshold = max(val)-(max(val)/3)
        # print "max: %d, threshold: %d" %(max(val),threshold)
        j = 0
        while j < len(val):
            if val[j] < threshold:
                # print "deleting value at %d", i
                del val[j]
            j += 1
        # print len(val)

        z.append(sum(val)/len(val))
        x.append(azOff)
        y.append(elOff)
    i = i+1
    print "%d: az: %d el: %d" %(i,azOff, elOff)
    azOld = azOff
print x
print y
print z

###########
# pour tracer le spectre:
###########
# remove outer 42 elements to match array dimensions and ditch the first and last measurements
# i = 0
# for i in range(0, len(z)):
#     j=0
#     while j < 42:
#         del z[i][j]
#         del z[i][len(z[i])-j-1]
#         j += 1
#
# print z
# convert lists to numpy arrays (for the plots)
xi = numpy.asarray(x)
yi = numpy.asarray(y)
zi = numpy.asarray(z)

xi = numpy.unique(xi)
yi = numpy.unique(yi)

print len(xi)
print len(yi)
print len(zi)

X,Y = numpy.meshgrid(xi,yi)
Z=zi.reshape(len(yi),len(xi))


plt.clf()
plt.pcolormesh(X, Y, Z) # here xi and yi are the vectors containing the offsets and zi contains the stream.
#plt.show()
plt.colorbar()
# normalize the zi array for the scatter plot
# maxLen = zi.max()
# zi /= maxLen
# for val in numpy.nditer(zi, op_flags=['readwrite']):
#     if val < 0:
#         val[...] = 0
#     elif val > 1:
#         val[...] = 1

plt.scatter(X, Y, c = Z, s = 100, vmin = Z.min(), vmax = Z.max())
plt.contour(X, Y, Z, 1000)
plt.xlabel( 'azimuth')
plt.ylabel( 'elevation')
plt.title( 'Solar flow map')

plt.show()
# filenametmp=date + ".png"
# print "backup file", filenametmp
# plt.savefig (filenametmp, format = 'png')
