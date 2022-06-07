#include <iostream>
#include <vector>
#include <../include/networklib.h>

using namespace std;

#define NETWORK_SIZE 50

int main(){

    cout << "Uniform degree distribution" << endl;
    Network net = Network(NETWORK_SIZE);
    net.getUniformDegreeSequence(1, 10);
    net.matchStubs();
    //net.printNetwork();

    //cout << endl << "Percolation: " << endl;
    vector<int> sr = net.nodePercolation();
    /*for(int m: sr){
        cout << m << ' ';
    }
    cout << endl;*/

    vector<float> result = net.computeGiantClusterPlot(20, sr);
    for(float i: result){
        cout << i << ' ';
    }

    cout << endl << "all done" << endl;
}