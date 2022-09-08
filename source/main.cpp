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
#include <algorithm>
#include <random>
#include <numeric>
#include <../include/degreedist.h>
#include <../include/networklib.h>
#include <../include/percolationlib.h>

using namespace std;

#define CONFIG_FILE_PATH "./experiments/test.yaml"
#define PLOT_BINS 51

vector<string> parseGiantCompConfigFile(string path, char delimiter=':', int params_num=5){
    string line;
    vector<string> params_value(params_num);
    ifstream myfile (path);

    if (myfile.is_open()){
        for(int i=0; i<params_num; i++){
            getline(myfile, line);
            //cout << line << endl;
            string str = line.substr(line.find(':')+1, line.size());
            //str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
            params_value[i] = str;
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file";
    } 
    return params_value;
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

vector<int> getDegDist(int nodes, char net_type, float param1){
    DegreeDistribution deg_dist = DegreeDistribution(nodes);
    switch(net_type){
        case 'b':
            deg_dist.generateBinomialDD(param1);
            break;
        case 'p':
            deg_dist.generatePowerLawDD(param1);
            break;
        case 'g':
            deg_dist.generateGeometricDD(param1);
            break;
        case 'f':
            deg_dist.generateFixedDD((int) param1);
            break;
        default:
            cout << "degree distribution not recognized" << endl;
            break;
    }
    return deg_dist.getDegreeDistribution();
}

vector<vector<double>> loadBinomialPMF(string path){
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
    return result;
}

double average(vector<double> data){
    double sum = (double) data.size()-count(data.begin(), data.end(), 0);
    if(sum == 0){
        return 0;
    }
    return reduce(data.begin(), data.end()) / sum; //(double) data.size();//
}

vector<vector<int>> transpose(vector<vector<int>> data){
    vector<vector<int>> result(data[0].size(), vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

vector<double> computeBinomialAverage(vector<vector<int>> data, string bin_path){
    vector<vector<double>> bin_pmf = loadBinomialPMF(bin_path);
    vector<vector<int>> sr_mat_t = transpose(data);
    vector<double> avg_sr;
    for(int i=0; i<sr_mat_t.size(); i++){
        vector<double> sr_row_t_double(sr_mat_t[i].begin(), sr_mat_t[i].end());
        avg_sr.push_back(average(sr_row_t_double));
    }

    vector<double> result;
    for(int i=0; i<PLOT_BINS; i++){
        double tmp = 0;
        for(int r=0; r<avg_sr.size(); r++){
            tmp += bin_pmf[i][r]*avg_sr[r];
        }
        result.push_back(tmp);
    }
    return result;
}

void percolation(){
    vector<string> exp_params = parseGiantCompConfigFile(CONFIG_FILE_PATH);

    int runs = stoi(exp_params[0]);
    int network_size = stoi(exp_params[1]);
    char network_type = exp_params[2][1];
    char percolation_type = exp_params[3][1];
    float param1 = stof(exp_params[4]);
    cout << "runs: " << runs << ", size: " << network_size << ", type: " << network_type << ", percolation: " << percolation_type << ", p1: " << param1 << endl;

    Network* net;
    Percolation* perc;
    vector<vector<int>> raw;
    vector<double> result;
    string binomial_data_path = "./data/pmf/binomial/binomialPMF_n"+to_string(network_size)+"_b"+to_string(PLOT_BINS)+".csv";
    switch(percolation_type){
    case 'n':
        for(int i=0; i<runs; i++){
            cout << "run: " << i << endl;
            net = new Network(getDegDist(network_size, network_type, param1));
            perc = new Percolation(net->getNetwork(), net->getEdgeList());
            raw.push_back(perc->UniformNodeRemoval());
            delete net, perc;
        }
        result = computeBinomialAverage(raw, binomial_data_path);
        saveResults("./results/raw/percolation_result.csv", result);
        break;
    
    default:
        cout << "percolation type not recognized" << endl;
        break;
    }

    /*
    if(percolation_type == 't' || percolation_type == 'f' || percolation_type == 'c' || percolation_type == 'e'){
        result = gcs.computeAverageGiantClusterSizeAsFunctionOfK();
        for(double r: result){
            cout << r << ", ";
        }
        cout << endl;

    }
    else if(percolation_type == 'l'){
        int m = network_size*param1*param2*0.5;
        gcs.loadBinomialPMF("./data/pmf/binomial/binomialPMF_n"+to_string(m)+"_b"+to_string(PLOT_BINS)+".csv");
        result = gcs.computeAverageGiantClusterSize(PLOT_BINS);
    }
    else if(percolation_type == 's'){
        //gcs.loadBinomialPMF("./data/pmf/binomial/binomialPMF_n"+to_string(network_size)+"_b"+to_string(PLOT_BINS)+".csv");
        result = gcs.computeBinomialAverage(PLOT_BINS);
    }
    else{
        gcs.loadBinomialPMF("./data/pmf/binomial/binomialPMF_n"+to_string(network_size)+"_b"+to_string(PLOT_BINS)+".csv");
        result = gcs.computeAverageGiantClusterSize(PLOT_BINS);
    }
    saveResults("./results/raw/node_perc_giant_cluster_exp_"+to_string(i)+".csv", result);
    */
}

int main(){
    /*
    Network net = Network("./data/results/1930s/edgelist_level_0.txt");
    net.removeSelfMultiEdges();
    //net.printNetwork();
    net.linkPercolation();
    vector<int> result = net.getSasfunctionofF(100);
    for(int r: result){
        cout << r << ", ";
    }
    */

    
    
    percolation();
    
    cout << endl << "all done" << endl;
    return 0;
}