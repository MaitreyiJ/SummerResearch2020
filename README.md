# SummerResearch2020
This repo contains some of the code that I developed while working on spike sorting algorithms as a part of my research in summer 2020.
This repo will be continuouly updated until August 2020. 
Following is the guide to read this repository-
1.SpikeForestTool-contains the SpikeForest Repo clonned 
2.spikeInterface-contains the modules for the package spikeInterface
3.Benchmarking contains the python code for benchmarking Mountain Sort, SpyKing Circus and MATLAB Algorithm form uNTT paper
         -matAlgorithm.py-imports and loads the file"resultFile.mat" not included here (due to size issues) from uNIT code, to benchmark it with other algorithms
         -FinalScriptforBenchmarking.py-This is the main script that was used to benchmark MountainSort,SpyKingCircus and MATLAB code from uNIT Paper using spikeInterface
         -Step1Thresholding.py-A sample implementation of spike sorting pipeline from https://towardsdatascience.com/whos-talking-using-k-means-clustering-to-sort-neural-events-in-python-e7a8a76f316
         
4.Plots-Figures generated during spike sorting benchmarking and comparision complete guide to figures available at https://docs.google.com/document/d/1lcIUoHKvgH8uC-ZFSPf3dkEODP7uts9i9vM84AmQXpY/edit?usp=sharing
