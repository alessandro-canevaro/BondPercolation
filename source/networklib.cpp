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

void Network::generateUniformDegreeSequence(int a, int b){
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

void Network::generateBinomialDegreeSequence(int n, float p){
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

void Network::generatePowerLawDegreeSequence(float alpha){

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

void Network::nodePercolation(){
    random_device rd;
    mt19937 gen(rd());

    vector<int> node_order;
    for(int i=0; i<nodes; i++){
        node_order.push_back(i);
    }
    shuffle(node_order.begin(), node_order.end(), gen);

    vector<int> results;
    int cluster_count = 0, max = 0;
    int r = 1;
    vector<vector<int>> clusters;
    for(int n: node_order){

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
        r++;

        max = 0;
        for(auto cls: clusters){
            if(cls.size() > max){
                max = cls.size();
            }
        }
        results.push_back(max);
    }

    sr = results;
}

vector<int> Network::getSr(){
    return sr;
}


GiantCompSize::GiantCompSize(){
    vector<vector<int>> sr_mat;
}

void GiantCompSize::generateNetworks(int net_num, int net_size, char type, float param1, float param2){
    vector<vector<int>> sr_matrix; 
    for(int i=0; i<net_num; i++){
        Network net = Network(net_size);
        switch (type){
        case 'u':
            net.generateUniformDegreeSequence((int) param1, (int) param2);
            break;
        case 'b':
            net.generateBinomialDegreeSequence((int) param1, param2);
            break;
        case 'p':
            net.generatePowerLawDegreeSequence((int) param1);
            break;
        }
        net.matchStubs();
        net.nodePercolation();
        sr_matrix.push_back(net.getSr());
    }
    sr_mat = sr_matrix;
}

vector<double> GiantCompSize::computeAverageGiantClusterSize(int bins){
    vector<vector<int>> sr_mat_t = this->transpose(sr_mat);

    vector<double> avg_sr;
    for(int i=0; i<sr_mat_t.size(); i++){
        avg_sr.push_back(this->average(sr_mat_t[i]));
    }

    vector<double> result;
    for(int i=0; i<bins; i++){
        result.push_back(this->computeGiantClusterSize(i/(float)bins, avg_sr));
    }
    return result;
}

double GiantCompSize::computeGiantClusterSize(float phi, vector<double> sr){
    vector<double> pmf = this->getBinomialPMF(phi, sr.size());

    double result = 0;
    for(int r=1; r<sr.size(); r++){
        result += pmf[r]*sr[r-1];
    }
    return result;
}

vector<double> GiantCompSize::getBinomialPMF(float phi, int nodes){
    vector<double> pmf(nodes + 1, 0.0);
    pmf[0] = 1.0;

    auto k = 0;
    for (auto n = 0; n < nodes; n++) {
        for (k = n + 1; k > 0; --k) {
        pmf[k] += phi * (pmf[k - 1] - pmf[k]);
        }
        pmf[0] *= (1 - phi);
    }
    return pmf;
}

vector<vector<int>> GiantCompSize::transpose(vector<vector<int>> data){
    vector<vector<int> > result(data[0].size(), vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

double GiantCompSize::average(vector<int> data){
    return reduce(data.begin(), data.end()) / (double)data.size();
}