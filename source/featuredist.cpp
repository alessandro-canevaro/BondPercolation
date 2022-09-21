/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.cpp
* @author Alessandro Canevaro
* @version 11/06/2022
*/

#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <../include/featuredist.h>

using namespace std;

FeatureDistribution::FeatureDistribution(vector<vector<int>> edges){
    edge_list = edges;
    num_edges = edge_list.size();
}

void FeatureDistribution::generateFeatureDist(int mu){
    random_device rd;
    mt19937 gen(rd());
    vector<int> feat;
    poisson_distribution<> d(mu);
    for(int i=0; i<num_edges; i++){
        feat.push_back(d(gen));
    }
    features = {feat};
}

void FeatureDistribution::generateCorrFeatureDist(){
    random_device rd;
    mt19937 gen(rd());
    this->getNet();

    vector<int> feat;
    int k1, k2;
    for(int i=0; i<num_edges; i++){
        k1 = net[edge_list[i][0]].size();
        k2 = net[edge_list[i][1]].size();
        poisson_distribution<> d((k1+k2));
        feat.push_back(d(gen));
    }
    features = {feat};
}

void FeatureDistribution::generateTemporalFeatureDist(int mu){
    this->generateFeatureDist(mu);
    this->ComputeTemporalFeature(30, 0, 20, 10.0, 10.0);
    vector<vector<int>> result;

    vector<int> feat = this->getFeatures(0);

    for(int i=0; i<num_edges; i++){
        if(feat[i] > 20){
            result.push_back(functions[20]);
        }
        else{
            result.push_back(functions[feat[i]]);
        }
    }
    features = this->transpose(result);
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
            double f = A*sin(2*3.1415*j/30.0 + phi)+k;
            //cout << "F0: " << i << " t: " << j << " f: " << round(f) << endl;
            row.push_back((int) f);
        }
        func.push_back(row);
    }
    functions = func;
}

vector<vector<int>> FeatureDistribution::transpose(vector<vector<int>> data){
    vector<vector<int>> result(data[0].size(), vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}