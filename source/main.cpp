#include <iostream>
#include <fstream>
#include <vector>
#include <../include/networklib.h>

using namespace std;

#define RUNS 10
#define NETWORK_SIZE 1000
#define NETWORK_TYPE 'p'
#define PARAM1 2.5
#define PARAM2 100

int main(){
    GiantCompSize gcs = GiantCompSize();
    gcs.generateNetworks(RUNS, NETWORK_SIZE, NETWORK_TYPE, PARAM1, PARAM2);
    vector<double> result = gcs.computeAverageGiantClusterSize(50);

    ofstream results_file("./results/node_perc_giant_cluster.csv");
    if(results_file.is_open()){
        results_file << RUNS << ',' << NETWORK_SIZE << ',' << NETWORK_TYPE << ',' << PARAM1 << ',' << PARAM2 << endl;

        for(int i=0; i<result.size()-1; i++){
            results_file << result[i] << ',';
            cout << result[i] << ", ";
        }
        results_file << result[result.size()-1];
        cout << result[result.size()-1];

        results_file.close();
    }
    else{
        cout << "Unable to open file.";
    }

    cout << endl << "all done" << endl;
}