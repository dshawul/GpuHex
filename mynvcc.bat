nvcc hex.cu -o hex --maxrregcount=64 -arch=sm_11 --ptxas-options=-v -use_fast_math -ccbin "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin" -Xcompiler /openmp 