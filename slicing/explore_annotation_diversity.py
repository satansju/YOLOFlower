from slicing import read_yolo
import glob
from tqdm import tqdm
import pandas as pd
from math import log
from plotnine import *
from matplotlib import pyplot as plt
from datar.all import *

def tab_yolo(path: str, ind: int) -> pd.DataFrame:
    ann = read_yolo(path)
    tab = {i : 0 for i in range(5)}
    for i in ann:
        tab[int(i[0])] += 1

    return pd.DataFrame({
                "path" : path,
                "Bud" : tab[0],
                "Flower" : tab[1],
                "Withered" : tab[2],
                "Immature" : tab[3],
                "Mature" : tab[4]
            }, index=[ind])

all_ann_paths = glob.glob("../Sliced/labels/*.txt")[0:100]

with tqdm(all_ann_paths, total = len(all_ann_paths)) as t:
  all_tab = pd.concat([tab_yolo(i, ind=ind) for ind, i in enumerate(t)])


# for path, tab in all_ann.items():
#     print("Path:", path)
#     print("Tab:", tab)
#     print(pd.DataFrame({
#         "path" : path,
#         "Bud" : tab[0],
#         "Flower" : tab[1],
#         "Withered" : tab[2],
#         "Immature" : tab[3],
#         "Mature" : tab[4]
#     }))
#     break

def ShannonEntropy(arr: list) -> float:
    norm_arr = [i/sum(arr) for i in arr]
    
    ShannonEntropy = -sum([i*log(i) if i != 0 else 0 for i in norm_arr])

    logCount = log(sum(arr))

    return ShannonEntropy*logCount if ShannonEntropy > 0 else 0

for ind, i in all_tab.iterrows():
    print(i[1:].array)
    break


all_tab["ShannonEntropy"] = [ShannonEntropy(i[1:].array) for ind, i in all_tab.iterrows()]

p = (as_tibble(all_tab) >>
        pivot_longer(["Bud","Flower","Withered","Immature","Mature"], names_to="class",values_to="n") >>
        ggplot(aes("ShannonEntropy", "n", color = "class"))
        # + geom_point(size = .1)
        + geom_smooth())


p.draw()

plt.show(block = True)