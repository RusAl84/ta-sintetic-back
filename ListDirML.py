from os import listdir
from os.path import isfile, join
mypath = "d:/ml/chat/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in onlyfiles:
    print(mypath+f)