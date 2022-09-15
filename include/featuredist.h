/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

class FeatureDistribution{
    public:
        FeatureDistribution(vector<vector<int>> edges);

        void generateFeatureDist(int mu);

        void generateCorrFeatureDist();

        void generateTemporalFeatureDist(int mu);

        vector<int> getFeatures(int t=0);
        
    protected:

        void getNet();
        void ComputeTemporalFeature(int max_t, int min_f, int max_f, double A, double k);
        vector<vector<int>> transpose(vector<vector<int>> data);

        int num_edges;
        vector<vector<int>> net;
        vector<vector<int>> edge_list;
        vector<vector<int>> features;
        vector<vector<int>> functions;

};