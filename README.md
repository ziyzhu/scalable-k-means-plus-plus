# Scalable-K-Means-plus-plus

[Stanford Scalable Kmeans||](https://arxiv.org/abs/1203.6402) implementation in C++

# Files

-   ./datasets: test datasets
-   ./generator.py: python script used to generate test datasets
-   ./kmeans_parallel.cpp: contains both parallelized k-means|| and kmeans++ implementation
-   ./kmeans_serial.cpp: sequential k-means clustering implementation

# Requirements

-   at least gcc-5.4.0
-   TBB

## build

```
make
```

## run

```
make test // modify ./Makefile to change test dataset
```
