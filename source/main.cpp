#include <iostream>
#include <vector>
#include <../include/networklib.h>

using namespace std;

#define NETWORK_SIZE 3

int main(){

    cout << "Uniform degree distribution" << endl;
    Network net = Network(NETWORK_SIZE);
    net.getUniformDegreeSequence(1, 4);
    net.matchStubs();
    net.printNetwork();

    cout << endl << "Percolation: " << endl;
    net.nodePercolation();
    /*
    cout << endl << "Binomial degree distribution" << endl;
    Network net2 = Network(NETWORK_SIZE);
    net2.getBinomialDegreeSequence(5, 0.5);
    net2.matchStubs();
    net2.printNetwork();

    cout << endl << "Percolation: " << endl;
    net2.nodePercolation();
    */
    cout << endl << "all done" << endl;
}