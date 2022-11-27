import numpy as np
import pandas as pd
from plotnine import *
import datar.all as r

res = pd.read_csv("runs/train/test_exp_0/results.csv")

res["gamma"] = [g for g in [0, .5, 1, 1.5, 2] for i in range(10)]

res.columns = res.columns.str.replace(' ', '')

print(r.as_tibble(res) >>
            r.pivot_longer(r._dplyr.across(~r.c(r.f.gamma, r.f.epoch)), names_to = "var", values_to="value"))

res_plot = (r.as_tibble(res) >>
            r.pivot_longer(r._dplyr.across(~r.c(r.f.gamma, r.f.epoch)), names_to = "var", values_to="value") >>
            r._dplyr.mutate(value = r._base.as_numeric(r.f.value)) >>
            ggplot(aes("epoch", "value", color="factor(gamma)", group="str(gamma)+var"))
 + geom_point()
 + facet_wrap("var", scales = "free_y")
 + theme(subplots_adjust={'wspace' : .5}))

res_plot.save("test_plot.png", width = 12.5, height = 10, dpi = 200)

