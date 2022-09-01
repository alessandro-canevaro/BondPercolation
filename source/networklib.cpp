/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.cpp
* @author Alessandro Canevaro
* @version 11/06/2022
*/

#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <../include/networklib.h>

using namespace std;

Network::Network(int n){
    nodes = n; // number of nodes
    vector<int> sequence;
    vector<vector<int>> network;
}

Network::Network(string path){
    vector<int> v1, v2, f;
    vector<vector<int>> order;
    ifstream myfile (path);
    int source, target, value;
    if (myfile.is_open()){
        while (myfile >> source >> target >> value){
            v1.push_back(source);
            v2.push_back(target);
            order.push_back({source, target});
            f.push_back(value);
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file";
    } 

    int m1 = *max_element(v1.begin(), v1.end());
    int m2 = *max_element(v2.begin(), v2.end());

    nodes = (m1 > m2) ? m1 : m2;
    nodes++; //count node 0
    vector<int> sequence;
    vector<vector<int>> net(nodes);

    for(int i=0; i<v1.size(); i++){
        net[v1[i]].push_back(v2[i]);
        net[v2[i]].push_back(v1[i]);
    }
    network = net;

    vector<int> indices(order.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return f[A] < f[B];
            });
    
    vector<vector<int>> sorted_order;
    vector<int> sorted_features;
    for(int idx: indices){
        //cout << order[idx][0] << " - " << order[idx][1] << "; F: " << f[idx] << endl;
        sorted_order.push_back(order[idx]);
        sorted_features.push_back(f[idx]);
    }

    feature_values = sorted_features;
    edge_order = sorted_order;
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

void Network::generateFixedDegreeSequence(int k){
    vector<int> degree_sequence (nodes);
    fill(degree_sequence.begin(), degree_sequence.end(), k);
    if((k*nodes) % 2 != 0){
        degree_sequence[0]++;
    }
    sequence = degree_sequence;
}

void Network::generateBinomialDegreeSequence(int n, float p){
    random_device rd;
    mt19937 gen(rd());
    binomial_distribution<> distrib(n, p);

    vector<int> degree_sequence (nodes);
    //cout << n << " " << p << " " << nodes << " " << n*p << endl;
    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = distrib(gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }
    //cout << "M = " << accumulate(degree_sequence.begin(), degree_sequence.end(), 0) / 2 << endl;
    sequence = degree_sequence;
}

void Network::generateGeometricDegreeSequence(float p){
    random_device rd;
    mt19937 gen(rd());
    geometric_distribution<int> distrib(p);

    vector<int> degree_sequence (nodes);
    //cout << n << " " << p << " " << nodes << " " << n*p << endl;
    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = distrib(gen);
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }
    //cout << "M = " << accumulate(degree_sequence.begin(), degree_sequence.end(), 0) / 2 << endl;
    sequence = degree_sequence;
}

void Network::generatePowerLawDegreeSequence(float alpha){
    random_device rd;
    mt19937 gen(rd());
    vector<double> intervals;
    vector<double> weights;
    //double c = 0;
    for(int i=2; i < (int) sqrt(nodes) +1; i++){
        intervals.push_back((double) i);
        weights.push_back(pow(i, -alpha));
        //c += pow(i, -alpha);
    }

    /*
    double mean = 0;
    for(int i=1; i < (int) sqrt(nodes) +1; i++){
        mean += i*pow(i, -alpha)/c;
    }
    cout << "theoretical mean: " << mean << endl;
    */
    piecewise_constant_distribution<double> distribution(intervals.begin(), intervals.end(), weights.begin());

    vector<int> degree_sequence (nodes);

    int sum = 1;
    while (sum % 2 != 0){ //generate a sequence until the total number of stubs is even
        for (int i=0; i<nodes; i++){
            degree_sequence[i] = (int) distribution(gen);
            //cout << degree_sequence[i] << ',';
        }
        sum = accumulate(degree_sequence.begin(), degree_sequence.end(), 0);
    }
    
    sequence = degree_sequence;
}

float Network::getDegreeDistMean(){
    double avg = 0;
    vector<int> dist(int((pow(nodes, 0.5)+1)));
    for(vector<int> n: network){
        dist[n.size()] ++;
        avg += n.size();
    }
    //for(int i: dist){
     //   cout << i << ",";
    //}
    //cout << endl;
    //cout << "size " << network.size() << endl;
    return avg/network.size();// - reduce(sequence.begin(), sequence.end()) / (float)sequence.size();
}

float Network::getSecondOrderMoment(){
    float mean = this->getDegreeDistMean();
    float variance = 0;
    for(vector<int> n: network){
        variance += pow(n.size()-mean, 2);
    }
    variance = variance / (float) network.size();
    cout << variance << endl;
    return variance + pow(mean, 2);
}

void Network::matchStubs(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, nodes-1);

    vector<int> degree_sequence = sequence; 
    vector<vector<int>> net (nodes);

    //cout << "matched started" << endl;
    for(int i=0; i<nodes; i++){
        //cout << "node " << i << endl;
        //cout << "left stubs: " << accumulate(degree_sequence.begin(), degree_sequence.end(), 0) << endl;
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
    //sequence = degree_sequence;
    /*
    shuffle(net.begin(), net.end(), gen);
    for(int i=0; i<net.size(); i++){
        shuffle(net[i].begin(), net[i].end(), gen);
    }
    */
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

void Network::printNetwork(){
    for(int i=0; i<nodes; i++){
        cout << "node " << i << " is connected to: ";
        for(auto j: network[i])  cout << j << ' ';
        cout << endl;
    }
}

void Network::generateUniformOrder(){
    random_device rd;
    mt19937 gen(rd());

    vector<int> order;
    for(int i=0; i<nodes; i++){
        order.push_back(i);
    }
    shuffle(order.begin(), order.end(), gen);

    node_order = order;
}

void Network::generateHighestDegreeFirstOrder(){
    random_device rd;
    mt19937 gen(rd());

    vector<vector<int>> degree_list(nodes); //estimate of the highest degree
    vector<int> order;

    for(int i=0; i<network.size(); i++){
        degree_list[network[i].size()].push_back(i);
    }

    for(int i=0; i<degree_list.size(); i++){
        if(degree_list[i].size() == 0){
            continue;
        }
        vector<int> nodes = degree_list[i];
        shuffle(nodes.begin(), nodes.end(), gen);
        order.insert(order.end(), nodes.begin(), nodes.end());
    }

    node_order = order;
}

vector<int> Network::getSasfunctionofK(int max_degree){
    vector<int> result(max_degree);
    fill(result.begin(), result.end(), 0);
    int previous_k = network[node_order[0]].size();
    result[previous_k] = sr[0];
    //cout << previous_k << " - " << sr[0] << endl;
    for(int i = 1; i<nodes; i++){
        if(previous_k < network[node_order[i]].size()){
            previous_k = network[node_order[i]].size();
            //cout << previous_k << " - " << sr[i] << endl;
            result[previous_k] = sr[i];
        }
    }
    return result;
}

void Network::nodePercolation(bool small_comp){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, nodes);

    vector<int> labels(nodes);
    fill(labels.begin(), labels.end(), -1);

    vector<vector<int>> net(nodes); //contains nodes
    vector<int> cluster_size(nodes); //contains labels to size
    fill(cluster_size.begin(), cluster_size.end(), 0);
    int new_label = 0;
    int giant_cluster_label = new_label;

    vector<int> result;
    result.push_back(0); //phy=0 -> no nodes
    int max_size = 1;

    vector<double> small_result;
    small_result.push_back(0);

    int node_count = 0;

    for(int n: node_order){
        //cout << "node: " << n << endl;

        labels[n] = new_label;
        cluster_size[new_label]++;
        new_label++;
        for(int n2: network[n]){
            if(labels[n2] != -1){
                //cout << "adding edge between node " << n << " and " << n2 << endl;
                net[n].push_back(n2);
                net[n2].push_back(n);

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
                        /*cout << "nodes in frontier: ";
                        for (int f: frontier){
                            cout << f <<" ";
                        }
                        cout << endl;*/

                        if(labels[frontier.back()] != lab){
                            //cout << "changing stuff" << endl;
                            labels[frontier.back()] = lab;
                            frontier.insert(frontier.begin(), net[frontier.back()].begin(), net[frontier.back()].end());
                        }
                        frontier.pop_back();
                    }
                }
            }
        }

        /*
        vector<double> ps;
        for(int i=1; i<11; i++){
            int val = count(cluster_size.begin(), cluster_size.end(), i);
            //cout << "found " << val << " cluster with size " << 1 << " ps: " << val*1/10000.0 << endl;
            ps.push_back(val*i/(double) nodes);   
        }
        double ps_mean = 0;
        for(int i=1; i<11; i++){
            ps_mean += ps[i-1]*i;
        }
        ps_mean = ps_mean / accumulate(ps.begin(), ps.end(), 0.0);
        small_result.push_back(ps_mean);
        */
        /*
        int val = count(cluster_size.begin(), cluster_size.end(), 1);
        small_result.push_back(val*1/(double) nodes);
        */
        //cout << "max size: " << max_size << endl;
        result.push_back(max_size);
    }

    double tot = 0;
    for(int i=1; i<1001; i++){
        int val = count(cluster_size.begin(), cluster_size.end(), i);
        //cout << "found " << val << " cluster with size " << i << " ps: " << val*i/100000.0 << endl;
        tot += val*i/(double) (100000.0-max_size);
        small_result.push_back(val*i/(double) (nodes-max_size));// << ", ";
    }
    cout << "total " << tot << endl;
    

    if(!small_comp){
        ss = small_result;
    }
    else{
        sr = result;
    }
}

int Network::getGiantClusterSize(){
    vector<int> visited(nodes);
    fill(visited.begin(), visited.end(), 0);
    int pos = 0;
    int c = 0; //number of clusters
    int n; //number of nodes in cluster
    int max_size = 0;

    while(true){
        auto found = find(visited.begin(), visited.end(), 0);
        if(found == visited.end()){
            break;
        }
        pos = found-visited.begin();

        vector<int> frontier;
        frontier.push_back(pos);
        n=0;
        c++;

        while(frontier.size() > 0){
            if(visited[frontier.back()] == 0){
                visited[frontier.back()] = c;
                n++;
                frontier.insert(frontier.begin(), network[frontier.back()].begin(), network[frontier.back()].end());
            }
            frontier.pop_back();
        }
        if(n > max_size){
            max_size = n;
        }
    }
    //cout << "c: " << c << endl;
    //cout << "n: " << n << endl;
    return max_size;
}

vector<vector<int>> Network::getNetwork(){
    return network;
}

vector<int> Network::getSr(){
    return sr;
}

vector<double> Network::getSs(){
    return ss;
}

bool Network::isConnected(){

    vector<int> visited(nodes);
    fill(visited.begin(), visited.end(), 0);
    int pos = 0;
    int c = 0;
    int n;

    while(true){
        auto found = find(visited.begin(), visited.end(), 0);
        if(found == visited.end()){
            break;
        }
        pos = found-visited.begin();

        vector<int> frontier;
        frontier.push_back(pos);
        n=0;
        c++;

        while(frontier.size() > 0){
            if(visited[frontier.back()] == 0){
                visited[frontier.back()] = c;
                n++;
                frontier.insert(frontier.begin(), network[frontier.back()].begin(), network[frontier.back()].end());
            }
            frontier.pop_back();
        }
        /*
        if(n>10){
            cout << "c: " << c << " size: " << n << endl;
        }
        */
    }
    //cout << "c: " << c << endl;
    //cout << "n: " << n << endl;
    return nodes == n;
}

vector<double> Network::getNeighborDegreeAvg(){
    vector<double> degree(int(pow(nodes, 0.5))+1);
    vector<double> degree_count(int(pow(nodes, 0.5))+1);
    fill(degree.begin(), degree.end(), 0.0);
    fill(degree_count.begin(), degree_count.end(), 0.0);
    
    for(int n=0; n<network.size(); n++){
        double avg = 0;
        degree_count[network[n].size()]++;
        for(int i=0; i<network[n].size(); i++){
            avg += network[network[n][i]].size();
        }
        degree[network[n].size()] += avg / (double) network[n].size();
    }
    for(int i=0; i<degree.size(); i++){
        //cout << degree[i] << " -> " << degree_count[i] << endl;
        if (degree_count[i] == 0){
            degree[i] = 0.0;
        }
        else{
            degree[i] = degree[i] / degree_count[i];
        }
    }

    return degree;
}

void Network::equalizeEdgeNumber(int M){
    //cout << "target M " << M << endl;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, nodes-1);

    int current_M = 0;
    vector<vector<int>> net = network;
    for(vector<int> n: network){
        current_M += n.size();
    }
    current_M = current_M / 2;
    int n1, n2;

    while(current_M > M){
        n1 = distrib(gen);
        if(net[n1].size() == 0){
            continue;
        }
        n2 = net[n1][0];
        net[n1].erase(net[n1].begin() + 0);
        net[n2].erase(find(net[n2].begin(), net[n2].end(), n1));
        current_M--;
    }

    while(current_M < M){
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
        current_M++;
    }

    network = net;

    current_M = 0;
    for(vector<int> n: network){
        current_M += n.size();
    }
    current_M = current_M / 2;
    //cout << "M after: " << current_M << endl;
}

void Network::linkPercolation(){
    vector<int> labels(nodes);
    fill(labels.begin(), labels.end(), -1);

    vector<vector<int>> net(nodes); //contains nodes
    vector<int> cluster_size(nodes); //contains labels to size
    fill(cluster_size.begin(), cluster_size.end(), 0);
    int new_label = 0;

    vector<int> result;
    result.push_back(0); //phy=0 -> no nodes
    int max_size = 2;

    for(vector<int> e: edge_order){ //edge_order_temporal
        net[e[0]].push_back(e[1]);
        net[e[1]].push_back(e[0]);
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
                    frontier.insert(frontier.begin(), net[frontier.back()].begin(), net[frontier.back()].end());
                }
                frontier.pop_back();
            }
        
        }
        //cout << "max size " << max_size << endl;
        result.push_back(max_size);
    }

    se = result;
}

vector<int> Network::getSe(){
    return se;
}

void Network::generateUniformEdgeOrder(){
    random_device rd;
    mt19937 gen(rd());

    vector<vector<int>> order;
    for(int n1 = 0; n1<network.size(); n1++){
        for(int n2: network[n1]){
            if(n2 > n1){
                order.push_back({n1, n2});
            }
        }
    }
    shuffle(order.begin(), order.end(), gen);
    edge_order = order;
    //for(vector<int> e: edge_order){
    //    cout << e[0] << " - " << e[1] << endl;
    //}
}

void Network::generateFeatureEdgeOrder(){
    random_device rd;
    mt19937 gen(rd());

    vector<vector<int>> order;
    for(int n1 = 0; n1<network.size(); n1++){
        for(int n2: network[n1]){
            if(n2 > n1){
                order.push_back({n1, n2});
            }
        }
    }
    shuffle(order.begin(), order.end(), gen);

    vector<int> features;
    poisson_distribution<> d(8);
    //binomial_distribution<> d(20, 0.5);
    for(int i=0; i<order.size(); i++){
        features.push_back(d(gen));
    }

    vector<int> indices(order.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return features[A] < features[B];
            });
    
    vector<vector<int>> sorted_order;
    vector<int> sorted_features;
    for(int idx: indices){
        //cout << order[idx][0] << " - " << order[idx][1] << "; F: " << features[idx] << endl;
        sorted_order.push_back(order[idx]);
        sorted_features.push_back(features[idx]);
    }

    feature_values = sorted_features;
    edge_order = sorted_order;

}

void Network::generateCorrFeatureEdgeOrder(){
    random_device rd;
    mt19937 gen(rd());

    vector<vector<int>> order;
    vector<int> tmp;
    for(int n1 = 0; n1<network.size(); n1++){
        for(int n2: network[n1]){
            if(n2 > n1){
                tmp = {n1, n2};
                shuffle(tmp.begin(), tmp.end(), gen);
                order.push_back(tmp);
            }
        }
    }
    //cout << order[300][0] <<" " <<order[300][1] << endl;
    shuffle(order.begin(), order.end(), gen);
    //cout << order[300][0] <<" " <<order[300][1] << endl;
    vector<int> features;
    int k1, k2;

    /*
    vector<double> intervals;
    vector<double> weights = {0.00298132, 0.0147078, 0.0408772, 0.0911621, 0.12548, 0.160859, 0.160859, 0.138797, 0.106135, 0.0711541, 0.0411422, 0.0231218, 0.0122565, 0.00642639, 0.00258381, 0.000795018, 0.00046376, 0.00006625155, 0.000132503};
    for(int i=2; i < 20 +1; i++){
        intervals.push_back((double) i);
        //weights.push_back(pow(i, -alpha));
        //c += pow(i, -alpha);
    }
    piecewise_constant_distribution<double> distribution(intervals.begin(), intervals.end(), weights.begin());
    */

    for(int i=0; i<order.size(); i++){
        k1 = network[order[i][0]].size();
        k2 = network[order[i][1]].size();
        poisson_distribution<> d((k1+k2));
        features.push_back(d(gen));
        //features.push_back(k1+k2);//(int) distribution(gen));
        //cout << "k1 " << k1 << ", k2 " << k2 << ", F: " << 50/(k1+k2) << endl;
    }
    
    /*
    for(int i=0; i<30; i++){
        int counter = count(features.begin(), features.end(), i);
        cout << "feature "<< i << ": " << counter/(double) features.size() << endl;
    }
    */
    
    vector<int> indices(order.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return features[A] < features[B];
            });
    
    vector<vector<int>> sorted_order;
    vector<int> sorted_features;
    for(int idx: indices){
        //cout << order[idx][0] << " - " << order[idx][1] << "; k1: " << network[order[idx][0]].size() << "; k2: " << network[order[idx][1]].size()<< "; F: " << features[idx] << endl;
        sorted_order.push_back(order[idx]);
        sorted_features.push_back(features[idx]);
    }

    feature_values = sorted_features;
    edge_order = sorted_order;
}

vector<vector<int>> Network::GenerateTemporalFeatures(){
    vector<vector<int>> functions;
    int max_t = 10;
    int min_f = 0;
    int max_f = 20;
    double A = 10.0;
    double k = 10.0;

    for(int i=min_f; i<=max_f; i++){
        vector<int> func;
        for(int j=0; j<=max_t; j++){
            double phi = asin((i-k)/A);
            double f = A*sin(2*3.1415*j/10.0 + phi)+k;
            //cout << "F0: " << i << " t: " << j << " f: " << round(f) << endl;
            func.push_back((int) f);
        }
        functions.push_back(func);
    }
    return functions;
}

vector<vector<int>> Network::ExtendTemporalFeatures(){
    vector<vector<int>> func = this->GenerateTemporalFeatures();
    vector<vector<int>> result;

    for(int i=0; i<feature_values.size(); i++){
        //cout << "F0: " << feature_values[i] << " extended: ";
        //for(int j: func[feature_values[i]]){
        //    cout << j << ", ";
        //}
        //cout << endl;
        result.push_back(func[feature_values[i]]);
    }
    return result;
}

vector<vector<int>> Network::GetTemporalEdgeOrder(int t){
    vector<vector<int>> extended = this->ExtendTemporalFeatures();
    vector<int> new_features;
    for(int i =0; i<extended.size(); i++){
        new_features.push_back(extended[i][t]);
    }

    vector<int> indices(new_features.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
           [&](int A, int B) -> bool {
                return new_features[A] < new_features[B];
            });
    
    vector<vector<int>> sorted_order;
    vector<int> sorted_features;
    for(int idx: indices){
        //cout << edge_order[idx][0] << " - " << edge_order[idx][1] << "; F: " << new_features[idx] << endl;
        sorted_order.push_back(edge_order[idx]);
        sorted_features.push_back(new_features[idx]);
    }
    
    //feature_values = sorted_features;
    edge_order_temporal = sorted_order;
    return sorted_order;
}

vector<int> Network::getSasfunctionofF(int max_F){
    vector<int> result(max_F);
    fill(result.begin(), result.end(), 0);
    int previous_f = feature_values[0];
    result[previous_f] = se[0];
    //cout << previous_f << " - " << se[0] << " (i: " << 0 << ")" << endl;
    for(int i = 1; i<se.size(); i++){
        if(previous_f < feature_values[i]){
            previous_f = feature_values[i];
            //cout << previous_f << ", ";
            if(previous_f >= max_F){break;}
            //cout << previous_f << " - " << se[i] << " (i: " << i << ")" << endl;
            result[previous_f] = se[i];
        }
    }

    //zero hold
    for(int i=1; i<result.size(); i++){
        if(result[i] == 0){
            result[i] = result[i-1];
        }
    }

    //cout << "done" << endl;
    return result;
}

GiantCompSize::GiantCompSize(){
    vector<vector<int>> sr_mat;
    vector<vector<int>> sk_mat;
}

void GiantCompSize::printNeighborDegreeAvg(int net_num, int net_size, char type, float param1, float param2){
    vector<vector<double>> result;
    float avg_k2_k = 0;
    for(int i=0; i<net_num; i++){
        Network net = Network(net_size);
        net.generatePowerLawDegreeSequence(param1);
        net.matchStubs();
        net.removeSelfMultiEdges();
        result.push_back(net.getNeighborDegreeAvg());
        avg_k2_k += net.getSecondOrderMoment()  / net.getDegreeDistMean();
    }
    avg_k2_k = avg_k2_k / net_num;
    cout << "avg k2 k2 = " << avg_k2_k << endl;

    vector<vector<double>> result_t = this->transpose(result);
    vector<double> avg_result;
    for(int i=0; i<result_t.size(); i++){
        avg_result.push_back(this->average(result_t[i]));
    }

    for(double r: avg_result){
        cout << r << ",";
    }
    cout << endl;
}

void GiantCompSize::generateNetworks(int net_num, int net_size, char type, char percolation_type, float param1, float param2){
    vector<vector<int>> sr_matrix; 
    vector<vector<int>> sk_matrix;
    vector<vector<double>> ss_matrix;
    double avgsize = 0;
    
    for(int i=0; i<net_num; i++){
        cout << i << endl;
        Network net = Network(net_size);
        switch (type){
        case 'u':
            net.generateUniformDegreeSequence((int) param1, (int) param2);
            break;
        case 'b':
            net.generateBinomialDegreeSequence((int) param1, param2);
            break;
        case 'p':
            net.generatePowerLawDegreeSequence(param1);
            break;
        case 'f':
            net.generateFixedDegreeSequence(param1);
            break;
        case 'g':
            net.generateGeometricDegreeSequence(param1);
            break;
        }
        net.matchStubs();
        net.removeSelfMultiEdges();
        //cout << net.getNetwork().size() << endl;
        //cout << "average degree: " << net.getDegreeDistMean() << endl;
        //cout << "second order moment:" << net.getSecondOrderMoment() << endl;
        //cout << i << endl;
        //avgsize += net.getGiantClusterSize();

        if(percolation_type == 't'){
            net.generateHighestDegreeFirstOrder();
            net.nodePercolation();
            sk_matrix.push_back(net.getSasfunctionofK(20));
        }
        else if(percolation_type == 'l'){
            net.equalizeEdgeNumber(param1*param2*net_size*0.5);
            //net.printNetwork();
            net.generateUniformEdgeOrder();
            net.linkPercolation();
            sr_matrix.push_back(net.getSe());
        }
        else if(percolation_type == 'f'){
            net.generateFeatureEdgeOrder();
            net.linkPercolation();
            sk_matrix.push_back(net.getSasfunctionofF(20));
        }
        else if(percolation_type == 'e'){
            net.generateFeatureEdgeOrder();
            int t = 9;
            vector<vector<int>> tmp = net.GetTemporalEdgeOrder(t);
            //for(int i=0; i<tmp.size(); i++){
            //    cout << tmp[i][0] << " - " << tmp[i][1] << endl;
            //}
            net.linkPercolation();
            sk_matrix.push_back(net.getSasfunctionofF(20));
        }
        else if(percolation_type == 'c'){
            //net.equalizeEdgeNumber(param1*param2*net_size*0.5);
            net.generateCorrFeatureEdgeOrder();
            net.linkPercolation();
            sk_matrix.push_back(net.getSasfunctionofF(20));
        }
        else if(percolation_type == 's'){
            net.generateUniformOrder();
            net.nodePercolation();
            ss_matrix.push_back(net.getSs());
        }
        else{
            net.generateUniformOrder();
            net.nodePercolation();
            sr_matrix.push_back(net.getSr());
        }

    }
    //cout << "avg max size: " << avgsize / net_num << endl;
    sk_mat = sk_matrix;
    sr_mat = sr_matrix;
    ss_mat = ss_matrix;
}

vector<double> GiantCompSize::computeAverageGiantClusterSizeAsFunctionOfK(){
    vector<vector<int>> sk_t = this->transpose(sk_mat);
    vector<double> result;
    for(vector<int> i: sk_t){
        result.push_back(this->average(i, false));
    }
    return result;
}

vector<double> GiantCompSize::computeAverageGiantClusterSize(int bins){
    vector<vector<int>> sr_mat_t = this->transpose(sr_mat);
    vector<double> avg_sr;
    for(int i=0; i<sr_mat_t.size(); i++){
        avg_sr.push_back(this->average(sr_mat_t[i], true));
    }

    vector<double> result;
    for(int i=0; i<bins; i++){
        double tmp = 0;
        for(int r=0; r<avg_sr.size(); r++){
            tmp += bin_pmf[i][r]*avg_sr[r];
        }
        result.push_back(tmp);
    }
    return result;
}

vector<double> GiantCompSize::computeBinomialAverage(int bins){
    vector<vector<double>> ss_mat_t = this->transpose(ss_mat);
    vector<double> avg_ss;
    for(int i=0; i<ss_mat_t.size(); i++){
        avg_ss.push_back(this->average(ss_mat_t[i]));
    }
    return avg_ss;
    vector<double> result;
    for(int i=0; i<bins; i++){
        double tmp = 0;
        for(int r=0; r<avg_ss.size(); r++){
            //cout << tmp << ", " <<bin_pmf[i][r] << ", "<< avg_ss[r] << ", "<< bin_pmf[i][r]*avg_ss[r] << endl;
            tmp += bin_pmf[i][r]*avg_ss[r];
        }
        result.push_back(tmp);
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

void GiantCompSize::loadBinomialPMF(string path){
    cout << "loading pmf: " << path << endl;
    vector<vector<double>> result;
    string line;
    ifstream myfile (path);

    if (myfile.is_open()){;
        while (getline(myfile, line)){
            //cout << "reading line: " << result.size() << endl;
            vector<double> row;
            while(true){
                auto pos = line.find(',');
                if(pos != string::npos){
                    string str = line.substr(0, pos);
                    line.erase(0, pos+1);
                    //cout << str << endl;
                    try{
                        row.push_back(stod(str));}
                        catch (out_of_range){
                            cout << "out of range " << str << endl;
                        }
                }
                else{
                    try{
                        row.push_back(stod(line));}
                        catch (out_of_range){
                            cout << "out of range " << line << endl;
                    }
                    break;
                }
            }
            result.push_back(row);
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file" << endl;
    }
    bin_pmf = result;
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
    vector<vector<int>> result(data[0].size(), vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

vector<vector<double>> GiantCompSize::transpose(vector<vector<double>> data){
    vector<vector<double>> result(data[0].size(), vector<double>(data.size()));
    for (vector<double>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<double>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

double GiantCompSize::average(vector<int> data, bool all){
    if (all){
        return reduce(data.begin(), data.end()) / (double)data.size();
    }
    else{
        double sum = (double) data.size()-count(data.begin(), data.end(), 0);
        if(sum == 0){
            return 0;
        }
        return reduce(data.begin(), data.end()) / sum;//data.size();
    }
}

double GiantCompSize::average(vector<double> data){
    double sum = (double) data.size()-count(data.begin(), data.end(), 0);
    if(sum == 0){
        return 0;
    }
    return reduce(data.begin(), data.end()) / sum; //(double) data.size();//
}