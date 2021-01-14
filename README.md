# learning-cuda
Learning CUDA programming

This repository contains my solutions to exercises of the course Parallel Programming (first semester, 2020 - 2021) at VNU-HCM, University of Science, lectured by Mr. Trung-Kien Tran.

## Exercise 1: CUDA Basics
### Part 1: Convert RGB image to gray-scale image
### Part 2: Blur image by convolution with Gaussian filter

## Exercise 2: How code is executed in CUDA
### Part 1: Array (sum) reduction
### Part 2: Calculate sum of two arrays using streams

## Exercise 3: Types of memory in CUDA
### Task: The same as Part 2 of Exercise 1
### kernel 1: using only GMEM
### kernel 2: copy the image from GMEM to SMEM, filter is still on GMEM
### kernel 3: copy the image from GMEM to SMEM and the filter from GMEM to CMEM

## Exercise 4: Parallelizing Radix Sort
### Task: Paralellize Radix Sort using CUDA
### Parallelized Bits Extraction: simply divide the arrays into blocks and attract bits on those
### Parallelized Scanning (Prefix Sum) on Bits: load the extracted bits into SMEM then do scanning
### Parallelized Ranking and Result writing: load the scanned array into SMEM then compute rank and write result

[More details will be updated soon...]
