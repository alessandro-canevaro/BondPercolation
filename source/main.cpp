#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <../include/networklib.h>
using namespace std;

void getUniformDegreeSequence(vector<int> &sequence, int a, int b){
    int n = sequence.size();

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(a, b);

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<n; i++){
            sequence[i] = distrib(gen);
        }
        sum = accumulate(sequence.begin(), sequence.end(), 0);
    }
}

void matchStubs(vector<vector<int>> &network, vector<int> sequence){
    int n = sequence.size();

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, n-1);

    for(int i=0; i<n; i++){
        while(sequence[i] > 0){
            //chose another (valid) node at random
            int node = -1;
            while(true){
                node = distrib(gen);
                if (node == i){ //self-edge case: we can stop if the node has at least 2 stubs
                    if (sequence[node] > 1){
                        break;
                    }
                }
                else{ //we can stop if the node has at least 1 stub
                    if (sequence[node] > 0){
                        break;
                    }
                }
            }

            //put the match in the two vectors of network
            network[i].push_back(node);
            network[node].push_back(i);

            //decrease the degree of the two nodes in the sequence
            sequence[i]--;
            sequence[node]--;
        }
    }
}

void printNetwork(vector<vector<int>> network){
    int n = network.size();
    for(int i=0; i<n; i++){
        cout << "node " << i << " is connected to: ";
        for(auto j: network[i])  cout << j << ' ';
        cout << endl;
    }
}

int main(){
    int n = 4; // number of nodes

    //generate degree sequence
    vector<int> sequence (n);

    getUniformDegreeSequence(sequence, 1, 4);
    for(int i=0; i<n; i++){
        cout << sequence[i] << ' ';
    }
    cout << endl;

    int sum = accumulate(sequence.begin(), sequence.end(), 0);
    cout << "the sum of elements is: " << sum << endl;

    //match nodes
    vector<vector<int>> network (n);

    matchStubs(network, sequence);

    printNetwork(network);

    Network net = Network(3);
    cout << "all done" << endl;
}