""" DBSCAN - Weather Station Clustering """
import numpy as np
import pandas as pd
from IPython.display import display

pdf = pd.read_csv("weather-stations20140101-20141231.csv")
#Remove rows with no value in "Tm" field
pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
display(pdf.head(5))

#############################################################################################
#DATA VISUALISATION
#############################################################################################
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
#Basemap using for plotting 2D data maps - provides transformations does not plot
#Basemap contains 25 different map projections
rcParams['figure.figsize'] = (14,10)
#Specification of position on Earth
llon = -140
ulon = -50
llat = 40
ulat = 65
pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) & (pdf['Lat'] < ulat) ]
my_map = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=ulon, urcrnrlat=ulat)
#min long. / lat.: llcrnrlon / llcrnrlat
#max long. / lat.: urcrnrlon / urcrnrlat

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

xs, ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm'] = xs.tolist()
pdf['ym'] = ys.tolist()

for _, row in pdf.iterrows():
    my_map.plot(row.xm, row.ym, markerfacecolor=([1,0,0]), marker='o', markersize=5, alpha=0.75)
plt.savefig("DBS_visualisation.pdf")

#############################################################################################
#CLUSTERING OF STATIONS BASED ON LOCATION
#############################################################################################
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm', 'ym' ]]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"] = labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

#Sample of clusters
#display(pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].tail(10))
#Outliers have a cluster label of -1
set(labels)

#############################################################################################
#VISUALISATION OF CLUSTERS BASED ON LOCATION
#############################################################################################
rcParams['figure.figsize'] = (14,10)
my_map = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=ulon, urcrnrlat=ulat)
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

for clust_number in set(labels):
    c = (([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)] )
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker = 'o', s = 20, alpha = 0.85)
    if clust_number != -1:
        cenx = np.mean(clust_set.xm)
        ceny = np.mean(clust_set.ym)
        plt.text(cenx, ceny, str(clust_number), fontsize=25, color='k',)
        print('Cluster '+str(clust_number)+', Avg Temp: '+str(np.mean(clust_set.Tm)) )
#plt.savefig("DBS_ClustLoc.pdf")
#plt.show()
#COLOURING ISNT CURRENTLY WORKING

#############################################################################################
#VISUALISATION OF CLUSTERS BASED ON LOCATION & TEMP
#############################################################################################
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels
realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

rcParams['figure.figsize'] = (14,10)
my_map = Basemap(projection='merc', resolution = 'l', area_thresh = 1000.0, llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=ulon, urcrnrlat=ulat)
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))
plt.show()

