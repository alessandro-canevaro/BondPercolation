/**
* Main file for runnning experiments
* @file main.cpp
* @author Alessandro Canevaro
* @version 11/06/2022
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <../include/networklib.h>

using namespace std;

#define CONFIG_FILE_PATH "./experiments/config.yaml"
#define PLOT_BINS 50

vector<vector<string>> parseGiantCompConfigFile(string path, char delimiter=':'){
    vector<vector<string>> result;
    string line;
    ifstream myfile (path);

    if (myfile.is_open()){
        getline(myfile, line);
        if(line == "giant_component:"){
            cout << "Experiments on giant component" << endl;
            while (getline(myfile, line)){
                vector<string> params_value(5);
                for(int i=0; i<5; i++){
                    getline(myfile, line);
                    //cout << line << endl;
                    string str = line.substr(line.find(':')+1, line.size());
                    //str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
                    params_value[i] = str;
                }
                result.push_back(params_value);
            }
        }
        else{
            cout << "no experiments on giant compoinent found" << endl;
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file";
    } 
    return result;
}

void saveResults(string path, vector<double> data){
    ofstream results_file(path);
    if(results_file.is_open()){
        for(int i=0; i<data.size()-1; i++){
            results_file << data[i] << ',';
            //cout << result[i] << ", ";
        }
        results_file << data[data.size()-1];
        //cout << result[result.size()-1];

        results_file.close();
        cout << "Result saved." << endl;
    }
    else{
        cout << "Unable to open file." << endl;;
    }
}

int main(){

    /*
    Network net = Network(10);
    net.generateBinomialDegreeSequence(10, 0.3);
    net.matchStubs();
    vector<vector<int>> graph = net.getNetwork();

    ofstream results_file("./results/raw/network.csv");
    if(results_file.is_open()){
        for(int i=0; i<graph.size(); i++){
            for(int j=0; j<graph[i].size()-1; j++){
                results_file << graph[i][j] << ',';
            }
            results_file << graph[i][graph[i].size()-1] << '\n';
        }
        results_file.close();
        cout << "Result saved." << endl;
    }
    else{
        cout << "Unable to open file." << endl;;
    }
    */

    vector<vector<string>> exp_params = parseGiantCompConfigFile(CONFIG_FILE_PATH);

    for(int i=0; i<exp_params.size(); i++){
        cout << endl << "Experiment: " << i << endl;

        int runs = stoi(exp_params[i][0]);
        int network_size = stoi(exp_params[i][1]);
        char network_type = exp_params[i][2][1];
        float param1 = stof(exp_params[i][3]);
        float param2 = stof(exp_params[i][4]);
        cout << "runs: " << runs << ", size: " << network_size << ", type: " << network_type << ", p1: " << param1 << ", p2: " << param2 << endl;

        GiantCompSize gcs = GiantCompSize();
        gcs.generateNetworks(runs, network_size, network_type, param1, param2);
        vector<double> result = gcs.computeAverageGiantClusterSize(PLOT_BINS);

        saveResults("./results/raw/node_perc_giant_cluster_exp_"+to_string(i)+".csv", result);
    }


    cout << endl << "all done" << endl;
    return 0;
}