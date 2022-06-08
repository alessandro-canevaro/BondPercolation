#include <iostream>
#include <fstream>
#include <vector>
#include <../include/networklib.h>

using namespace std;

#define NETWORK_SIZE 1000

int main(){

    cout << "Uniform degree distribution" << endl;
    Network net = Network(NETWORK_SIZE);
    net.getBinomialDegreeSequence(NETWORK_SIZE, 0.003);
    net.matchStubs();
    cout << "stubs matched" << endl;
    //net.printNetwork();

    //cout << endl << "Percolation: " << endl;
    vector<int> sr = net.nodePercolation();
    cout << "percolation completed" << endl;
    /*for(int m: sr){
        cout << m << ' ';
    }
    cout << endl;*/

    vector<float> result = net.computeGiantClusterPlot(20, sr);
    for(float i: result){
        cout << i << ' ';
    }

    ofstream results_file("./results/node_perc_giant_cluster.csv");
    if(results_file.is_open()){
        for(int i=0; i<result.size()-1; i++){
            results_file << result[i] << ',';
        }
        results_file << result[result.size()-1];

        results_file.close();
    }
    else{
        cout << "Unable to open file.";
    }

    cout << endl << "all done" << endl;
}