/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.cpp
* @author Alessandro Canevaro
* @version 11/06/2022
*/

#include <random>
#include <vector>
#include <../include/degreedist.h>

using namespace std;

DegreeDistribution::DegreeDistribution(int nodes){
    nodes = nodes;
    mt19937 rand_gen(rd());
}

void DegreeDistribution::generateBinomialDD(float p){
    vector<int> degree_sequence (nodes);
    binomial_distribution<> distrib(nodes, p);

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = distrib(rand_gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }
    degree_dist = degree_sequence;
}

void DegreeDistribution::geenrateFixedDD(int k){
    vector<int> degree_sequence (nodes);
    fill(degree_sequence.begin(), degree_sequence.end(), k);
    if((k*nodes) % 2 != 0){
        degree_sequence[0]++;
    }
    degree_dist = degree_sequence;
}

void DegreeDistribution::generateGeometricDD(float p){
    vector<int> degree_sequence (nodes);
    geometric_distribution<int> distrib(p);

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = distrib(rand_gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }
    degree_dist = degree_sequence;
}

void DegreeDistribution::generatePowerLawDD(float alpha){
    vector<int> degree_sequence (nodes);

    vector<double> intervals, weights;
    for(int i=2; i < (int) sqrt(nodes) +1; i++){
        intervals.push_back((double) i);
        weights.push_back(pow(i, -alpha));
    }

    piecewise_constant_distribution<double> distrib(intervals.begin(), intervals.end(), weights.begin());

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = (int) distrib(rand_gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }
    degree_dist = degree_sequence;
}

vector<int> DegreeDistribution::getDegreeDistribution(){
    return degree_dist;
}
