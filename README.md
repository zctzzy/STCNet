### Source code for our paper entitled "Deep Transfer Learning for Intelligent Wireless Traffic Prediction Based on Cross-Domain Big Data"
see our [paper](https://chuanting.github.io/pdf/ieee_jsac_2019.pdf) for more details.
### Requirements
System: Ubuntu 16.04 LTS with GeForce GTX TITAN X, 64 bit OS 

Python: 3.6.3 Anaconda 64-bit

Pytorch version: 0.4.0

CUDA Driver version: 384.130


### About the data
The original data is with precision of 15 decimal digits and the size is about 500 MB. To reduce the data size so that it can be downloaded by others, the data is rounded to 3 decimals. That is, the data is manipulated through
```
data = np.around(data, 3)
```
Besides, the original data has five channels, i.e., (sms_in, sms_out, call_in, call_out, internet). We grouped them as (sms, call, internet), that is, three channels.

The wireless traffic data is named as "data_git_version.7z", please unzip it to your local computer.

The crawled cross-domain data is named as "crawled_feature.csv", there are 4 columns, i.e., (social, BSs, POI1, POI2).

There is also a file named as "cluster_label_20.csv", this is the clustered results.

### How to execute
Download the code to your own computer, open the terminal, simply type:
```
python demo_three_cluster.py
```

You can also specify parameters to control the training, such as set "epoch_size" to 300

```
python demo_three_cluster.py -epoch_size 300
```

There are others parameters can be set, please see the code for details.


### Note
As the last week of this dataset contains the New Year's Eve, so for that specific data point, we do not predict it as it is a very challenging task to predict such "outliers". We use a simple linear prediction, that is, $y_t = (y_{t-1}+y_{t-2}+y_{t-3})/3$.