import pickle
import numpy as np

class Results():
    def __init__(self):
        self.x = 2


result_file = Results()
result_file.y = 2

saving = False

if saving:
    with open("someobject1.result", "wb") as output_file:
        pickle.dump(result_file, output_file)

else:
    file = pickle.load(open("someobject.result",'rb'))
    if hasattr(file, 'x'):
        print(file.x)
    if hasattr(file, 'y'):
        print(file.y)