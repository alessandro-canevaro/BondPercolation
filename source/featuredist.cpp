/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.cpp
* @author Alessandro Canevaro
* @version 11/06/2022
*/

#include <random>
#include <vector>
#include <algorithm>
#include <../include/featuredist.h>

using namespace std;

FeatureDistribution::FeatureDistribution(vector<vector<int>> edge_list){
    edge_list = edge_list;
    num_edges = edge_list.size();
    mt19937 rand_gen(rd());
}

void FeatureDistribution::generateFeatureDist(int mu){
    vector<int> feat;
    poisson_distribution<> d(mu);
    for(int i=0; i<num_edges; i++){
        feat.push_back(d(rand_gen));
    }
    features = {feat};
}

void FeatureDistribution::generateCorrFeatureDist(){
    this->getNet();

    vector<int> feat;
    int k1, k2;
    for(int i=0; i<num_edges; i++){
        k1 = net[edge_list[i][0]].size();
        k2 = net[edge_list[i][1]].size();
        poisson_distribution<> d((k1+k2));
        feat.push_back(d(rand_gen));
    }
    features = {feat};
}

void FeatureDistribution::generateTemporalFeatureDist(int mu){
    this->generateFeatureDist(mu);
    this->ComputeTemporalFeature(10, 0, 20, 10.0, 10.0);
    vector<vector<int>> result;

    vector<int> feat = this->getFeatures(0);

    for(int i=0; i<num_edges; i++){
        //cout << "F0: " << feature_values[i] << " extended: ";
        //for(int j: func[feature_values[i]]){
        //    cout << j << ", ";
        //}
        //cout << endl;
        result.push_back(functions[feat[i]]);
    }
    features = result;
}

vector<int> FeatureDistribution::getFeatures(int t){
    return features[t];
}

void FeatureDistribution::getNet(){
    vector<int> tmp;
    for(int i=0; i<num_edges; i++){
        tmp.push_back(max(edge_list[i][0], edge_list[i][1]));
    }
    int nodes = *max_element(tmp.begin(), tmp.end());
    nodes++; //count node 0

    vector<vector<int>> network(nodes);
    for(int i=0; i<num_edges; i++){
        network[edge_list[i][0]].push_back(edge_list[i][1]);
        network[edge_list[i][1]].push_back(edge_list[i][0]);
    }

    net = network;
}

void FeatureDistribution::ComputeTemporalFeature(int max_t, int min_f, int max_f, double A, double k){
    vector<vector<int>> func;

    for(int i=min_f; i<=max_f; i++){
        vector<int> row;
        for(int j=0; j<=max_t; j++){
            double phi = asin((i-k)/A);
            double f = A*sin(2*3.1415*j/10.0 + phi)+k;
            //cout << "F0: " << i << " t: " << j << " f: " << round(f) << endl;
            row.push_back((int) f);
        }
        func.push_back(row);
    }
    functions = func;
}