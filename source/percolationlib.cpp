/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.cpp
* @author Alessandro Canevaro
* @version 11/06/2022
*/

#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <../include/percolationlib.h>
#include <../include/networklib.h>
#include <../include/degreedist.h>
#include <../include/featuredist.h>


Percolation::Percolation(vector<vector<int>> network, vector<vector<int>> edge_list){
    net = network;
    edges = edge_list;
    nodes = network.size();
    feat_dist = new FeatureDistribution(edges);
}

vector<int> Percolation::UniformNodeRemoval(){
    vector<int> order(nodes);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rand_gen);

    this->nodePercolation(order);
    return perc_results;
}

vector<int> Percolation::HighestDegreeNodeRemoval(){
    vector<vector<int>> degree_list(nodes); //estimate of the highest degree
    vector<int> order;

    for(int i=0; i<nodes; i++){
        degree_list[net[i].size()].push_back(i);
    }

    for(int i=0; i<nodes; i++){
        if(degree_list[i].size() > 0){
            vector<int> nodes = degree_list[i];
            shuffle(nodes.begin(), nodes.end(), rand_gen);
            order.insert(order.end(), nodes.begin(), nodes.end());
        }
    }

    this->nodePercolation(order);
    return perc_results;
}

vector<int> Percolation::UniformEdgeRemoval(){
    vector<vector<int>> order = edges;
    shuffle(order.begin(), order.end(), rand_gen);

    this->edgePercolation(order);
    return perc_results;
}

vector<int> Percolation::FeatureEdgeRemoval(int mu){
    vector<vector<int>> order = edges;

    feat_dist->generateFeatureDist(mu);
    vector<int> features = feat_dist->getFeatures();

    vector<int> indices(order.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return features[A] < features[B];
            });
    
    vector<vector<int>> sorted_order;
    for(int idx: indices){
        sorted_order.push_back(order[idx]);
    }
    
    this->edgePercolation(sorted_order);
    return perc_results;    
}

vector<int> Percolation::CorrFeatureEdgeRemoval(){
    vector<vector<int>> order = edges;

    feat_dist->generateCorrFeatureDist();
    vector<int> features = feat_dist->getFeatures();

    vector<int> indices(order.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return features[A] < features[B];
            });
    
    vector<vector<int>> sorted_order;
    for(int idx: indices){
        sorted_order.push_back(order[idx]);
    }

    this->edgePercolation(sorted_order);
    return perc_results;    
}

vector<int> Percolation::TemporalFeatureEdgeRemoval(){
    vector<int> x;
    return x;
}

void Percolation::nodePercolation(vector<int> node_order){
    uniform_int_distribution<> distrib(0, nodes);

    vector<int> labels(nodes);
    fill(labels.begin(), labels.end(), -1);

    vector<vector<int>> old_net = net;
    vector<vector<int>> new_net(nodes); //contains nodes
    vector<int> cluster_size(nodes); //contains labels to size
    fill(cluster_size.begin(), cluster_size.end(), 0);
    int new_label = 0;
    int giant_cluster_label = new_label;

    vector<int> result;
    result.push_back(0); //phi=0 -> no nodes
    int max_size = 1;

    int node_count = 0;

    for(int n: node_order){
        //cout << "node: " << n << endl;
        labels[n] = new_label;
        cluster_size[new_label]++;
        new_label++;
        for(int n2: old_net[n]){
            if(labels[n2] != -1){
                //cout << "adding edge between node " << n << " and " << n2 << endl;
                new_net[n].push_back(n2);
                new_net[n2].push_back(n);

                if(labels[n] != labels[n2]){ //merge
                    //cout << "merging cluster " << labels[n] << " with cluster " << labels[n2] << endl; 
                    vector<int> frontier; //contains nodes
                    int lab;
                    if(cluster_size[labels[n]]>cluster_size[labels[n2]]){
                        frontier.push_back(n2);
                        lab = labels[n];                
                    }
                    else{
                        frontier.push_back(n);
                        lab = labels[n2];     
                    }

                    cluster_size[lab] += cluster_size[labels[frontier.back()]];
                    cluster_size[labels[frontier.back()]] = 0;
                    if(cluster_size[lab] > max_size){
                        max_size = cluster_size[lab];
                        giant_cluster_label = lab;
                    }

                    //cout << "lab: " << lab << endl;
                    while (frontier.size() > 0){
                        if(labels[frontier.back()] != lab){
                            //cout << "changing stuff" << endl;
                            labels[frontier.back()] = lab;
                            frontier.insert(frontier.begin(), new_net[frontier.back()].begin(), new_net[frontier.back()].end());
                        }
                        frontier.pop_back();
                    }
                }
            }
        }
        result.push_back(max_size);
    }
    perc_results = result;
}

void Percolation::edgePercolation(vector<vector<int>> edge_order){
    vector<int> labels(nodes);
    fill(labels.begin(), labels.end(), -1);

    vector<vector<int>> new_net(nodes); //contains nodes
    vector<int> cluster_size(nodes); //contains labels to size
    fill(cluster_size.begin(), cluster_size.end(), 0);
    int new_label = 0;

    vector<int> result;
    result.push_back(0); //phi=0 -> no nodes
    int max_size = 2;

    for(vector<int> e: edge_order){
        new_net[e[0]].push_back(e[1]);
        new_net[e[1]].push_back(e[0]);
        //cout << "considering edge between: " << e[0] << " - " << e[1] << endl;

        if(labels[e[0]] == -1 && labels[e[1]] == -1){ //both nodes are new
            //cout << "both nodes are not present, marking them with label: " << new_label << endl;
            labels[e[0]] = new_label;
            labels[e[1]] = new_label;
            cluster_size[new_label] += 2; //always smaller then max_size
            new_label++;
        } 
        else if(labels[e[0]] == -1){
            //cout << "label of node e[0]" << e[0] << " is -1; changed to " << labels[e[1]] << endl;
            labels[e[0]] = labels[e[1]];
            cluster_size[labels[e[1]]]++;

            if(cluster_size[labels[e[1]]] > max_size){
                max_size = cluster_size[labels[e[1]]];
            }
        }
        else if(labels[e[1]] == -1){
            //cout << "label of node e[1]" << e[1] << " is -1; changed to " << labels[e[0]] << endl;
            labels[e[1]] = labels[e[0]];
            cluster_size[labels[e[0]]]++;

            if(cluster_size[labels[e[0]]] > max_size){
                max_size = cluster_size[labels[e[0]]];
            }
        }
        else if(labels[e[0]] != labels[e[1]]){
            //relabel
            int lab;
            vector<int> frontier;
            if(cluster_size[labels[e[0]]] < cluster_size[labels[e[1]]]){
                lab = labels[e[1]];
                frontier.push_back(e[0]);
            }
            else{
                lab = labels[e[0]];
                frontier.push_back(e[1]);
            }
            //cout << "relabiling nodes from " << frontier.back() << " with label " << lab << endl; 

            cluster_size[lab] += cluster_size[labels[frontier.back()]];
            cluster_size[labels[frontier.back()]] = 0;
            if(cluster_size[lab] > max_size){
                max_size = cluster_size[lab];
            }

            while (frontier.size() > 0){
                if(labels[frontier.back()] != lab){
                    labels[frontier.back()] = lab;
                    frontier.insert(frontier.begin(), new_net[frontier.back()].begin(), new_net[frontier.back()].end());
                }
                frontier.pop_back();
            }
        
        }
        //cout << "max size " << max_size << endl;
        result.push_back(max_size);
    }
    perc_results = result;
}