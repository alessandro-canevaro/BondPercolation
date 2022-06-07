#include <iostream>
#include <vector>
#include <../include/networklib.h>
using namespace std;

int main(){

    Network net = Network(3);
    net.getUniformDegreeSequence(1, 4);
    net.matchStubs();
    net.printNetwork();

    cout << "all done" << endl;
}