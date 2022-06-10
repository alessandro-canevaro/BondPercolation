#include <iostream>
#include <fstream>
#include <vector>
#include <../include/networklib.h>

using namespace std;

int main(){
    GiantCompSize gcs = GiantCompSize();
    gcs.generateNetworks(5, 1000, 'b', 1000, 0.003);
    vector<double> result = gcs.computeAverageGiantClusterSize(50);

    ofstream results_file("./results/node_perc_giant_cluster.csv");
    if(results_file.is_open()){
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