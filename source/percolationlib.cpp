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
    random_device rd;
    mt19937 gen(rd());
    vector<int> order(nodes);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), gen);

    this->nodePercolation(order, false);
    
    return perc_results;
}

vector<double> Percolation::UniformNodeRemovalSmallComp(){
    random_device rd;
    mt19937 gen(rd());
    vector<int> order(nodes);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), gen);

    this->nodePercolation(order, true);
    
    return small_comp_results;
}

vector<int> Percolation::HighestDegreeNodeRemoval(int max_degree){
    random_device rd;
    mt19937 gen(rd());
    vector<vector<int>> degree_list(nodes); //estimate of the highest degree
    vector<int> order;

    for(int i=0; i<nodes; i++){
        degree_list[net[i].size()].push_back(i);
    }

    for(int i=0; i<nodes; i++){
        if(degree_list[i].size() > 0){
            vector<int> nodes = degree_list[i];
            shuffle(nodes.begin(), nodes.end(), gen);
            order.insert(order.end(), nodes.begin(), nodes.end());
        }
    }

    this->nodePercolation(order, false);
    
    vector<int> result(max_degree);
    fill(result.begin(), result.end(), 0);
    
    int previous_k = net[order[0]].size();
    result[previous_k] = perc_results[0];
    //cout << previous_k << " - " << perc_results[0] << endl;
    
    for(int i = 1; i<nodes; i++){
        //cout << "i: "<< i << ", " << net[order[i]].size() << endl;
        if(previous_k < net[order[i]].size()){
            previous_k = net[order[i]].size();
            //cout << previous_k << " - " << perc_results[i] << endl;
            if(previous_k>=max_degree){
                break;
            }
            result[previous_k] = perc_results[i];
        }
    }
    
    if(result[max_degree-1] == 0){
        result[max_degree-1] = perc_results[nodes];
    }

    int count_zeros = 0;
    for(int i=0; i<max_degree; i++){
        if(result[i]==0){
            count_zeros++;
        }
        else{
            break;
        }
    }

    for(int i=max_degree-1; i>count_zeros; i--){
        if(result[i-1] == 0){
            result[i-1] = result[i];
        }
    }
    
    return result;
    //return perc_results;
}

vector<int> Percolation::UniformEdgeRemoval(){
    random_device rd;
    mt19937 gen(rd());
    vector<vector<int>> order = edges;
    shuffle(order.begin(), order.end(), gen);

    this->edgePercolation(order);
    return perc_results;
}

vector<int> Percolation::FeatureEdgeRemoval(int mu, int max_feature){
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
    vector<int> sorted_features;
    for(int idx: indices){
        sorted_order.push_back(order[idx]);
        sorted_features.push_back(features[idx]);
    }
    
    this->edgePercolation(sorted_order);

    vector<int> result(max_feature);
    fill(result.begin(), result.end(), 0);
    
    int previous_f = sorted_features[0];
    result[previous_f] = perc_results[0];
    //cout << previous_f << " - " << perc_results[0] << endl;
    int edges = perc_results.size();
    for(int i = 1; i<edges; i++){
        //cout << "i: "<< i << ", " << net[order[i]].size() << endl;
        if(previous_f < sorted_features[i]){
            previous_f = sorted_features[i];
            //cout << previous_k << " - " << perc_results[i] << endl;
            if(previous_f>=max_feature){
                break;
            }
            result[previous_f] = perc_results[i];
        }
    }
    
    if(result[max_feature-1] == 0){
        result[max_feature-1] = perc_results[edges-1];
    }

    int count_zeros = 0;
    for(int i=0; i<max_feature; i++){
        if(result[i]==0){
            count_zeros++;
        }
        else{
            break;
        }
    }

    for(int i=max_feature-1; i>count_zeros; i--){
        if(result[i-1] == 0){
            result[i-1] = result[i];
        }
    }
    
    return result;
    //return perc_results;    
}

vector<int> Percolation::CorrFeatureEdgeRemoval(int max_feature){
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
    vector<int> sorted_features;
    for(int idx: indices){
        sorted_order.push_back(order[idx]);
        sorted_features.push_back(features[idx]);
    }

    this->edgePercolation(sorted_order);

    vector<int> result(max_feature);
    fill(result.begin(), result.end(), 0);
    
    int previous_f = sorted_features[0];
    result[previous_f] = perc_results[0];
    //cout << previous_f << " - " << perc_results[0] << endl;
    int edges = perc_results.size();
    for(int i = 1; i<edges; i++){
        //cout << "i: "<< i << ", " << net[order[i]].size() << endl;
        if(previous_f < sorted_features[i]){
            previous_f = sorted_features[i];
            //cout << previous_k << " - " << perc_results[i] << endl;
            if(previous_f>=max_feature){
                break;
            }
            result[previous_f] = perc_results[i];
        }
    }
    
    if(result[max_feature-1] == 0){
        result[max_feature-1] = perc_results[edges-1];
    }

    int count_zeros = 0;
    for(int i=0; i<max_feature; i++){
        if(result[i]==0){
            count_zeros++;
        }
        else{
            break;
        }
    }

    for(int i=max_feature-1; i>count_zeros; i--){
        if(result[i-1] == 0){
            result[i-1] = result[i];
        }
    }
    
    return result;
    //return perc_results;    
}

vector<int> Percolation::TemporalFeatureEdgeRemoval(int mu, int t, int max_feature){
    vector<vector<int>> order = edges;
    if(t==0){
        feat_dist->generateTemporalFeatureDist(mu);
    }
    vector<int> features = feat_dist->getFeatures(t);

    vector<int> indices(order.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return features[A] < features[B];
            });
    
    vector<vector<int>> sorted_order;
    vector<int> sorted_features;
    for(int idx: indices){
        sorted_order.push_back(order[idx]);
        sorted_features.push_back(features[idx]);
    }
    
    this->edgePercolation(sorted_order);

    vector<int> result(max_feature);
    fill(result.begin(), result.end(), 0);
    
    int previous_f = sorted_features[0];
    result[previous_f] = perc_results[0];
    //cout << previous_f << " - " << perc_results[0] << endl;
    int edges = perc_results.size();
    for(int i = 1; i<edges; i++){
        //cout << "i: "<< i << ", " << net[order[i]].size() << endl;
        if(previous_f < sorted_features[i]){
            previous_f = sorted_features[i];
            //cout << previous_k << " - " << perc_results[i] << endl;
            if(previous_f>=max_feature){
                break;
            }
            result[previous_f] = perc_results[i];
        }
    }
    
    if(result[max_feature-1] == 0){
        result[max_feature-1] = perc_results[edges-1];
    }

    int count_zeros = 0;
    for(int i=0; i<max_feature; i++){
        if(result[i]==0){
            count_zeros++;
        }
        else{
            break;
        }
    }

    for(int i=max_feature-1; i>count_zeros; i--){
        if(result[i-1] == 0){
            result[i-1] = result[i];
        }
    }
    
    return result;
    //return perc_results;    
}

void Percolation::nodePercolation(vector<int> node_order, bool small_comp){
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

    vector<double> small_result;
    //small_result.push_back(0);

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

    if(small_comp){
        double tot = 0;
        for(int i=1; i<40; i++){ //for phi=1 count how many small comp. of size i there are.
            int val = count(cluster_size.begin(), cluster_size.end(), i);
            //cout << "found " << val << " cluster with size " << i << " ps: " << val*i/100000.0 << endl;
            tot += val*i/(double) (nodes-max_size);
            small_result.push_back(val*i/(double) (nodes-max_size));// << ", ";
        }
        //cout << "total " << tot << endl;
        small_comp_results = small_result;
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