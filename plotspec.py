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
for spec in series:
    # date = spec["date"]
    # date = (date.rsplit(':',1))[0]
    # mode = spec["digital"]
    azOff = spec["azOff"]
    elOff = spec["elOff"]

    x.append(azOff)
    y.append(elOff)
    z.append(spec["value"])

print z

###########
# pour tracer le spectre:
###########
# remove outer 42 elements to match array dimensions and ditch the first and last measurements
i = 0
for i in range(0, len(z)):
    j=0
    while j < 42:
        del z[i][j]
        del z[i][len(z[i])-j-1]
        j += 1

print z
# convert lists to numpy arrays (for the plots)
xi = numpy.asarray(x)
yi = numpy.asarray(y)
zi = numpy.asarray(z)


plt.clf()
plt.pcolormesh(xi, yi, zi) # here xi and yi are the vectors containing the offsets and zi contains the stream.
plt.colorbar()
# normalize the zi array for the scatter plot
maxLen = zi.max()
zi /= maxLen
for val in numpy.nditer(zi, op_flags=['readwrite']):
    if val < 0:
        val[...] = 0
    elif val > 1:
        val[...] = 1

plt.scatter(xi, yi, c = zi, s = 100, vmin = zi.min(), vmax = zi.max())
plt.contour(xi, yi, zi, 300)
plt.xlabel( 'azimuth')
plt.ylabel( 'elevation')
plt.title( 'Solar flow map')

plt.show()
# filenametmp=date + ".png"
# print "backup file", filenametmp
# plt.savefig (filenametmp, format = 'png')
