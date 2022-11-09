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
#include <../include/networklib.h>

using namespace std;

Network::Network(vector<int> degree_sequence){
    nodes = degree_sequence.size();
    this->matchStubs(degree_sequence);
    this->removeSelfMultiEdges();
    this->makeEdgeList();
}

Network::Network(vector<vector<int>> edge_list){
    vector<int> tmp;
    for(int i=0; i<edge_list.size(); i++){
        tmp.push_back(max(edge_list[i][0], edge_list[i][1]));
    }
    nodes = *max_element(tmp.begin(), tmp.end()) + 1;//count node 0

    vector<vector<int>> net(nodes);

    for(int i=0; i<edge_list.size(); i++){
        net[edge_list[i][0]].push_back(edge_list[i][1]);
        net[edge_list[i][1]].push_back(edge_list[i][0]);
    }

    network = net;
    this->removeSelfMultiEdges();
    this->makeEdgeList();
}

void Network::equalizeEdges(int m){
    random_device rd;
    mt19937 gen(rd());
    vector<vector<int>> net = network;
    uniform_int_distribution<> distrib(0, nodes-1);

    int current_m = edge_list.size();
    int n1, n2;

    while(current_m > m){
        n1 = distrib(gen);
        if(net[n1].size() == 0){
            continue;
        }
        n2 = net[n1][0];
        net[n1].erase(net[n1].begin() + 0);
        net[n2].erase(find(net[n2].begin(), net[n2].end(), n1));
        current_m--;
    }

    while(current_m < m){
        while(true){
            n1 = distrib(gen);
            n2 = distrib(gen);
            if(n1 != n2){
                if(find(net[n1].begin(), net[n1].end(), n2) == net[n1].end()){
                    break;
                }
            }
        }
        net[n1].push_back(n2);
        net[n2].push_back(n1);
        current_m++;
    }

    network = net;

    this->makeEdgeList();
}

void Network::printNetwork(){
    for(int i=0; i<nodes; i++){
        cout << "node " << i << " is connected to: ";
        for(auto j: network[i])  cout << j << ' ';
        cout << endl;
    }
}

vector<vector<int>> Network::getEdgeList(){
    return edge_list;
}

vector<vector<int>> Network::getNetwork(){
    return network;
}

vector<double> Network::getDegDist(){
    vector<double> dist(nodes);
    for(int i=0; i<nodes; i++){
        dist[i] = 0;
    }
    for(int i=0; i<nodes; i++){
        dist[network[i].size()] += 1.0;
    }
    for(int i=0; i<nodes; i++){
        dist[i] = dist[i] / (double) nodes;
    }
    return dist;
}

void Network::matchStubs(vector<int> degree_sequence){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, nodes-1);

    vector<vector<int>> net(nodes);

    for(int i=0; i<nodes; i++){
        while(degree_sequence[i] > 0){
            //chose another (valid) node at random
            int node = -1;
            while(true){
                node = distrib(gen);
                
                if (degree_sequence[node] > 0){
                    break;
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
    network = net;
}

void Network::removeSelfMultiEdges(){
    vector<vector<int>> net(nodes);
    int removed = 0;
    for(int i=0; i<network.size(); i++){
        vector<int> node = network[i];
        sort(node.begin(), node.end());
        node.push_back(-1); //extra value
        for(int j=0; j<network[i].size(); j++){
            if(node[j] == i){
                //self loop
                removed++;
                continue;
            }
            if(node[j] == node[j+1]){
                //self loop or multiedge
                removed++;
                continue;
            }
            net[i].push_back(node[j]);
        }
    }
    //cout << "removed " << removed << " self or multi edges" << endl;
    network = net;
}

void Network::makeEdgeList(){
    vector<vector<int>> order;
    for(int n1 = 0; n1<nodes; n1++){
        for(int n2: network[n1]){
            if(n2 > n1){
                order.push_back({n1, n2});
            }
        }
    }
    edge_list = order;
}