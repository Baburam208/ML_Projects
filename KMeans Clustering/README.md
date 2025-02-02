## This repository contains implementation of KMeans clustering algorithms
I have tried to cluster NEPSE stocks based on the annual return and volatility. And the optimal `k` is found using
the elbow method.


### Usage
1. first run 'calculating features for k-means.py' file for generating the feature metrics for performing
    k-mean clustering.
2. run 'k_means_clustering.py' for performing KMeans clustering.


### Results
The elbow plot is as follows.

![alt text](<elbow method for optimal k.png>)

Finding the optimal `k` from the elbow plot seems quite confusing. So lets try to use `kneed` module from python.

### Using `kneed` module to find the optimal `k` value.
The result from the `kneed` is as follows.

![alt text](<Knee Point.png>)

This suggest us to use `k=4`.

### Final clustering result using `k=4`

![alt text](<clustering visualization.png>)


### Stocks in each clusters are as follows
Stocks in each cluster: <br />
Cluster 0: KRBL, ICFC, PFL, OHL, NGPL, NICL <br />
Cluster 1: ADBL, CZBIL, MBL, NBL, NICA, PCBL, SANIMA, BPCL, CHCL <br />
Cluster 2: EBL, NABIL, SBI, SBL, SCB, MNBBL, NLICL, NLIC, SICL <br />
Cluster 3: GBBL, JBBL, MDB, SADBL, SHINE, SHL, API, AHPC, SHPC, CIT, HIDCL, ALICL, LICN, NIL <br />
