import sys
sys.path.insert(1, 'C:\\Users\\AHMET\\Documents\\GitHub\\CS-Stuff\\STAT 3025Q')
import STAT3025_Tools.Stat_Tools as st

data = "13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 23, 25, 25, 25, 30, 33, 33, 33, 33, 35, 35, 36, 40, 45, 46, 52, 70"
data = [int(x) for x in data.split(", ")]

from ISLP import load_data
Hitters = load_data('Hitters')

print(Hitters.columns)