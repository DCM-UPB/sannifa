import subprocess
from pylab import *

def extract_time(line):
    return float(line.split(":")[1].strip().split()[0])

def getPlotY(data, X):
    Y = []
    for x in X:
        Y.append(data[x])
    return Y

def getRelativeY(Y, Yrel):
    ret = []
    for it, y in enumerate(Y):
        ret.append(y / Yrel[it])
    return ret


outputs = {}

print("Executing libqnets/poly benchmark...")
print()
outputs["lqpoly"] = subprocess.getoutput( "../build/benchmark/bench_qpoly" )
print("Done.")
print(outputs["lqpoly"])

print("Executing libqnets/templ benchmark...")
print()
outputs["lqtempl"] = subprocess.getoutput( "../build/benchmark/bench_qtempl" )
print("Done.")
print(outputs["lqtempl"])

print("Executing libtorch benchmark...")
print()
outputs["ltorch"] = subprocess.getoutput( "OMP_NUM_THREADS=1 ../build/benchmark/bench_torch" )
print("Done.")
print(outputs["ltorch"])

benchNames = ["small (6x12x12x1)", "medium (24x48x48x1)", "large (96x192x192x1)"]
derivNames = ["noderiv", "d1", "d1+d2", "vd1", "d1+vd1", "d1+d2+vd1"]
data = {}

for key in outputs.keys():
    data[key] = {}

    triggerLine = -8 # save the line number where trigger was found
    countBenchmark = -1
    for it, line in enumerate(outputs[key].split("\n")):
        if (line.strip() == "Time per Evaluation"):
            triggerLine = it
            countBenchmark += 1
            data[key][benchNames[countBenchmark]] = {}
        elif ((it > triggerLine + 1) and (it < triggerLine + 8)):
            data[key][benchNames[countBenchmark]][derivNames[it-triggerLine-2]] = extract_time(line)

print(data)            

plotDerivs = ["noderiv", "d1+vd1", "d1+d2+vd1"]
colors = ["orange", "blue", "red"]
pos = arange(len(plotDerivs))
nlibs = len(data)
print(nlibs)
height = 0.8 / nlibs 

reflib = "lqpoly"

figure()
#suptitle("ANN Derivatives Benchmark")
for itb, benchName in enumerate(benchNames):
    subplot(3, 1, itb+1)
    title(benchName + " FFNN")
    refY = getPlotY(data[reflib][benchName], plotDerivs)
    for itk, key in enumerate(data.keys()):
        datay = getPlotY(data[key][benchName], plotDerivs)
        ploty = getRelativeY(datay, refY)
        barh(pos-itk*height, ploty, height, log = True, color = colors[itk])
    yticks(pos - 0.5*nlibs*height, plotDerivs)
    if (itb == 2):
        xlabel("Time (relative to worst)")
    if (itb == 0):
        legend(data.keys())

tight_layout()
show()
