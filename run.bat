set OMP_NUM_THREADS=4
cmake --build .\build\ --config Release
.\source\main.exe
python .\visualization\percolation.py