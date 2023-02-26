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
#include <omp.h>
#include <../include/degreedist.h>
#include <../include/networklib.h>
#include <../include/percolationlib.h>
#include "../include/progressbar.hpp"

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

float getDegDistMean(int nodes, char net_type, float param1){
    float C = 0;
    float mean = 0;
    switch(net_type){
        case 'b':
            mean = nodes*param1;
            break;
        case 'p':
            for(int i=2; i < (int) sqrt(nodes) +1; i++){
                C += pow(i, -param1);
            }
            for(int i=2; i < (int) sqrt(nodes) +1; i++){
                mean += i * pow(i, -param1) / C;
            }
            break;
        case 'g':
            mean = (1-param1)/param1;
            break;
        case 'f':
            mean = param1;
            break;
        default:
            cout << "degree distribution not recognized" << endl;
            break;
    }
    return mean;
}

vector<vector<double>> loadBinomialPMF(string path){
    cout << "loading pmf: " << path << endl;
    vector<vector<double>> result(PLOT_BINS);
    vector<string> lines(PLOT_BINS);
    ifstream myfile (path, ios::in);
    progressbar bar(PLOT_BINS);

    if (myfile.is_open()){
        for(int i=0; i<PLOT_BINS; i++){
            getline(myfile, lines[i]);
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file" << endl;
    }
    //cout << "file has: " << lines.size() << " lines" << endl;

    #pragma omp parallel for num_threads(16)
    for(int i=0; i<PLOT_BINS; i++){
        //cout << "reading line: " << i << endl;
        string line = lines[i];
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
        result[i] = row;
        bar.update();
    }
    return result;
}

double average(vector<double> data){
    double sum = (double) data.size();//-count(data.begin(), data.end(), 0);
    if(sum == 0){
        return 0;
    }
    return accumulate(data.begin(), data.end(), 0.0) / sum; //(double) data.size();//
}

vector<vector<int>> transpose(vector<vector<int>> data){
    vector<vector<int>> result(data[0].size(), vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

vector<vector<double>> transpose(vector<vector<double>> data){
    vector<vector<double>> result(data[0].size(), vector<double>(data.size()));
    for (vector<double>::size_type i = 0; i < data[0].size(); i++) 
        for (vector<double>::size_type j = 0; j < data.size(); j++) {
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

vector<double> computeBinomialAverage(vector<vector<double>> data, string bin_path){
    vector<vector<double>> bin_pmf = loadBinomialPMF(bin_path);
    vector<vector<double>> sr_mat_t = transpose(data);
    vector<double> avg_sr;
    for(int i=0; i<sr_mat_t.size(); i++){
        avg_sr.push_back(average(sr_mat_t[i]));
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

vector<double> computeAverage(vector<vector<int>> data){
    vector<vector<int>> data_t = transpose(data);
    vector<double> result;
    for(vector<int> row: data_t){
        vector<double> row_double(row.begin(), row.end());
        result.push_back(average(row_double));
    }
    return result;
}

vector<double> computeAverage(vector<vector<double>> data){
    vector<vector<double>> data_t = transpose(data);
    vector<double> result;
    for(vector<double> row: data_t){
        result.push_back(average(row));
    }
    return result;
}

vector<vector<int>> loadEdgeList(string path){
    //cout << "loading" << endl;
    vector<vector<int>> edgelist;
    ifstream myfile(path);
    int v1, v2, f;
    if (myfile.is_open()){
        //cout << "file open" << endl;
        while (myfile >> v1 >> v2 >> f){
            //cout << v1 << endl;
            edgelist.push_back({v1, v2});
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file";
    } 
    return edgelist;
}

vector<int> loadFeatureList(string path){
    //cout << "loading" << endl;
    vector<int> featlist;
    ifstream myfile(path);
    int v1, v2, f;
    if (myfile.is_open()){
        //cout << "file open" << endl;
        while (myfile >> v1 >> v2 >> f){
            //cout << v1 << endl;
            featlist.push_back(f);
        }
        myfile.close();
    }
    else{
        cout << "Unable to open file";
    } 
    return featlist;
}

void percolation(){
    vector<string> exp_params = parseGiantCompConfigFile(CONFIG_FILE_PATH);

    int runs = stoi(exp_params[0]);
    int network_size = stoi(exp_params[1]);
    char network_type = exp_params[2][1];
    char percolation_type = exp_params[3][1];
    float param1 = stof(exp_params[4]);
    cout << "runs: " << runs << ", size: " << network_size << ", type: " << network_type << ", percolation: " << percolation_type << ", p1: " << param1 << endl;

    vector<vector<int>> raw(runs);
    vector<vector<int>> raw_corr(runs);
    vector<vector<int>> raw_uncorr(runs);
    vector<vector<double>> raw_deg_dist(runs);
    vector<vector<double>> raw_joint_dist(runs);
    vector<int> data, row;
    int m = round(network_size*0.5*getDegDistMean(network_size, network_type, param1));
    string binomial_data_path = "./data/pmf/binomial/binomialpmf_n"+to_string(network_size)+"_b"+to_string(PLOT_BINS)+".csv";
    
    progressbar bar(runs);

    vector<double> test;

    double t1 = omp_get_wtime();
    
    #pragma omp parallel for num_threads(24) schedule(dynamic)
    for(int i=0; i<runs; i++){
        Network net = Network(getDegDist(network_size, network_type, param1));

        raw_deg_dist[i] = net.getDegDist();

        Percolation perc = Percolation(net.getNetwork(), net.getEdgeList());

        if(percolation_type=='f'){
            raw[i] = perc.FeatureEdgeRemoval(8, 21);
        }
        if(percolation_type=='c'){
            raw_corr[i] = perc.CorrFeatureEdgeRemoval(21, true);
            raw_uncorr[i] = perc.CorrFeatureEdgeRemoval(21, false);
        }

        raw_joint_dist[i] = perc.getJointDistribution();

        bar.update();
    }
    cout << endl;

    double t2 = omp_get_wtime();
    cout << "percolation took: " << t2-t1 << " seconds" << endl;

    saveResults("./results/raw/perco_result_corr.csv", computeAverage(raw_corr));
    saveResults("./results/raw/perco_result_uncorr.csv", computeAverage(raw_uncorr));
    saveResults("./results/raw/perco_result_degdist.csv", computeAverage(raw_deg_dist));
    saveResults("./results/raw/perco_result_jointdist.csv", computeAverage(raw_joint_dist));

    double t3 = omp_get_wtime();
    cout << "data processing took: " << t3-t2 << " seconds" << endl;
}

int main(){   
    percolation();
    //tmp_net_perc();
    cout << endl << "all done" << endl;
    return 0;
}
