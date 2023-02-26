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

void FeatureDistribution::generateCorrFeatureDist(bool correlated){
    random_device rd;
    mt19937 gen(rd());
    this->getNet();
    vector<double> joint_dist(21*50); //max_feat * f(k, j)

    vector<int> feat;
    int curr_feat;
    int k1, k2;
    for(int i=0; i<num_edges; i++){
        k1 = net[edge_list[i][0]].size();
        k2 = net[edge_list[i][1]].size();
        poisson_distribution<> d(50/(k1+k2));
        curr_feat = d(gen);
        feat.push_back(curr_feat);
        joint_dist[curr_feat+21*(50/(k1+k2))] += 1;
    }
    //shuffle for uncorrelated case
    if (!correlated){
        random_shuffle(feat.begin(), feat.end());
    }

    features = {feat};

    //joint distribution
    double sum = accumulate(joint_dist.begin(), joint_dist.end(), 0.0);
    for(int i = 0; i < joint_dist.size(); i++){
        if (joint_dist[i] > 0.0){
            joint_dist[i] = joint_dist[i]/sum;
        }
    } 
    joint_distribution = joint_dist;
}

vector<double> FeatureDistribution::getJointDistribution(){
    return joint_distribution;
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

vector<int> FeatureDistribution::getFeatures(int t){
    return features[t];
}

vector<vector<int>> FeatureDistribution::transpose(vector<vector<int>> data){
    vector<vector<int>> result(data[0].size(), vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}