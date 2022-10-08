export OMP_NUM_THREADS=4
cd build/
cmake ..
make
cd ../
./main
