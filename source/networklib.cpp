#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <../include/networklib.h>

using namespace std;

Network::Network(int n){
    nodes = n; // number of nodes
    vector<int> sequence;
    vector<vector<int>> network;
}

void Network::getUniformDegreeSequence(int a, int b){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(a, b);

    vector<int> degree_sequence (nodes);

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = distrib(gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }

    sequence = degree_sequence;
}

void Network::getBinomialDegreeSequence(int n, float p){
    random_device rd;
    mt19937 gen(rd());
    binomial_distribution<> distrib(n, p);

    vector<int> degree_sequence (nodes);

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = distrib(gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }

    sequence = degree_sequence;
}

void Network::matchStubs(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, nodes-1);

    vector<int> degree_sequence = sequence;
    vector<vector<int>> net (nodes);

    for(int i=0; i<nodes; i++){
        while(degree_sequence[i] > 0){
            //chose another (valid) node at random
            int node = -1;
            while(true){
                node = distrib(gen);
                if (node == i){ //self-edge case: we can stop if the node has at least 2 stubs
                    if (degree_sequence[node] > 1){
                        break;
                    }
                }
                else{ //we can stop if the node has at least 1 stub
                    if (degree_sequence[node] > 0){
                        break;
                    }
                }
            }

            //put the match in the two vectors of network
            net[i].push_back(node);
            net[node].push_back(i);

            //decrease the degree of the two nodes in the sequence
            degree_sequence[i]--;
            degree_sequence[node]--;
        }
    }
    //sequence = degree_sequence;
    network = net;

}

void Network::printNetwork(){
    for(int i=0; i<nodes; i++){
        cout << "node " << i << " is connected to: ";
        for(auto j: network[i])  cout << j << ' ';
        cout << endl;
    }
}

vector<int> Network::nodePercolation(){
    random_device rd;
    mt19937 gen(rd());

    vector<int> node_order;
    for(int i=0; i<nodes; i++){
        node_order.push_back(i);
    }
    shuffle(node_order.begin(), node_order.end(), gen);

    vector<int> sr;
    int cluster_count = 0, max = 0;
    //int r = 1;
    vector<vector<int>> clusters;
    for(int n: node_order){
        //cout << "r: " << r << " considering node: " << n << endl;

        vector<int> merge;
        for(int e: network[n]){
            //cout << "considering edge to node: " << e << endl;
            for(int cls=0; cls<clusters.size(); cls++){
                auto found = find(begin(clusters[cls]), end(clusters[cls]), e);
                if(found != end(clusters[cls])){
                    //cout << "the node was found in cluster with label: " << cls << endl;
                    if(find(merge.begin(), merge.end(), cls) != merge.end()){
                        //cout << "element already in merge" << endl;
                    }
                    else{
                        merge.push_back(cls);
                    }
                    break;
                }
            }
        }

        vector<int> new_cls{n};
        clusters.push_back(new_cls);
        cluster_count++;
        //cout << "node: " << n << " is added in a new cluster. total: " << cluster_count << endl;
        sort(merge.begin(), merge.end(), greater<int>());
        for(int cls: merge){
            //cout << "cluster " << cls << " inserted into " << clusters.size()-1 << endl;
            clusters[clusters.size()-1].insert(clusters[clusters.size()-1].end(), clusters[cls].begin(), clusters[cls].end());
            clusters.erase(clusters.begin() + cls);
            cluster_count--;
        }
        //cout << "total clusters: " << cluster_count << endl;
        //r++;

        max = 0;
        for(auto cls: clusters){
            if(cls.size() > max){
                max = cls.size();
            }
        }
        sr.push_back(max);
    }

    return sr;
}

float Network::computeMeanSizeOfLargestCluster(float phi, vector<int> sr){
    float result = 0;
    for(int r=1; r<nodes; r++){
        //cout << "bin of " <<nodes << " and " << r << " is: " << binomialCoeff(nodes, r)*pow(phi, r)*pow(1-phi, nodes-r) << endl;
        result += binomialCoeff(nodes, r)*pow(phi, r)*pow(1-phi, nodes-r)*sr[r-1];
    }
    return result;
}

vector<float> Network::computeGiantClusterPlot(int bins, vector<int> sr){
    vector<float> result;
    for(int i=0; i<bins; i++){
        result.push_back(this->computeMeanSizeOfLargestCluster(i/(float)bins, sr));
    }
    return result;
}

long long int binomialCoeff(const int n, const int k) {
    std::vector<long long int> aSolutions(k);
    aSolutions[0] = n - k + 1;

    for (int i = 1; i < k; ++i) {
        aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
    }

    return aSolutions[k - 1];
}