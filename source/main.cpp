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

    cout << endl << "Binomial degree distribution" << endl;
    Network net2 = Network(NETWORK_SIZE);
    net.getBinomialDegreeSequence(5, 0.5);
    net.matchStubs();
    net.printNetwork();

    cout << endl << "all done" << endl;
}