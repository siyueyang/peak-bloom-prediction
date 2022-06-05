
![](https://badgen.net/badge/cherry-prediction/competition/pink) ![](https://badgen.net/badge/best-narrative/statistics/red) 

# Cherry blossom peak bloom prediction

A submission for 2022 [George Mason’s Department of Statistics cherry blossom peak bloom prediction competition](https://competition.statistics.gmu.edu/). We won the Award for Best Narrative (Statistics).


### Authors

- [@MiaoshiqiLiu](https://github.com/MiaoshiqiLiu)
- [@siyueyang](https://github.com/siyueyang)


## Cherry blossom prediction competition

The first cherry bloom competition focuses on the predictions for Washington, D.C. (USA), Kyoto (Japan), Liestal-Weideli (Switzerland) and Vancouver, BC (Canada).  

#### Competition data

The competition provides cleaned data in the [Github repo](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction), containing 

- **peak bloom dates** across three international locations: Kyoto, Liestal, and Washington D.C., while the bloom dates are _missing_ for Vancouver. 

- **peak bloom dates** for various sites across Switzerland, Japan, Sourth Korea, and the USA. 

- **global meteorological data** from the Global Historical Climatology Network (GHCN) in the `rnoaa` package with illustration in the initial analysis. 

## Our attempts

#### Data preparation

We extracted Vancouver bloom peak dates during 2004-2021 from the National Park Website. 
Below shows the distribution of the peak bloom dates.

![](img/bloom_peak_time.png)

We construct another feature "peak bloom days" since January 1st of the year untial peak bloom for every record. Peak bloom days in Kyoto are more
concentrated, while the days in other three locations are more spread out. Cherry blossom
is earlier in Vancouver and Washington DC than Kyoto and Liestal; this may due to the
differences in the locations and climate features.

![](img/peak_bloom_days.png)

The cherry blossom is highly related to the temperature, as evidenced from a seasonal advance of the cherry blossom associated with a distinctive increase in the global temperature.  We extracted weather data from `rnoaa` package. As weather data is missing across several years, we used the Kalman Smoothing time series model to impute the daily temperature. 


Here is the monthly average maximum and minimum temperature across four sies. Due to the different temperature trends, seperate models or hierarchical models should be considered for forecasting the bloom dates. We used separate models in our analysis. 

![](img/seasonal_temp.png)


Additionally, we summarised daily weather data into 

- accumulated growing degree days (AGDD)
- first growing days of year (FGDDY)
- last growing days of year (LGDDY) 
- accumulated freezing degree days (AFDD)
- first freezing days of year (AFDDY)
- last freezing days of year (LFDDY)
- average maximum temperature in Winter (Tmax-W)
- average maximum temperature in Spring (Tmax-S)
- average minimum temperature in Winter (Tmin-W)
- average minimum temperature in Spring (Tmin-S)
- average precipitation in Winter (PRCPW)
- average precipitation in Spring (PRCP-S)
