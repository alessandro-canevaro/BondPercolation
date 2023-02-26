/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

class FeatureDistribution;

class Percolation{

    public:
        Percolation(vector<vector<int>> network, vector<vector<int>> edge_list);

        vector<int> FeatureEdgeRemoval(int mu, int max_feature);

        vector<int> FeatureEdgeRemoval(vector<int> features, int max_feature);

        vector<int> CorrFeatureEdgeRemoval(int max_feature, bool correlated=true);

        vector<double> getJointDistribution();

    protected:

        void nodePercolation(vector<int> node_order, bool small_comp);
        void edgePercolation(vector<vector<int>> edge_order);

        int nodes;
        vector<vector<int>> net;
        vector<vector<int>> edges;
        FeatureDistribution* feat_dist;
        vector<int> perc_results;
        vector<int> small_comp_results;

        vector<double> joint_distribution;
};