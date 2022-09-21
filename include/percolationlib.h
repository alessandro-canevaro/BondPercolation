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

        vector<int> UniformNodeRemoval();

        vector<double> UniformNodeRemovalSmallComp();

        vector<int> HighestDegreeNodeRemoval(int max_degree);

        vector<int> UniformEdgeRemoval();

        vector<int> FeatureEdgeRemoval(int mu, int max_feature);

        vector<int> CorrFeatureEdgeRemoval(int max_feature);

        vector<int> TemporalFeatureEdgeRemoval(int mu, int t, int max_feature);

    protected:

        void nodePercolation(vector<int> node_order, bool small_comp);
        void edgePercolation(vector<vector<int>> edge_order);

        int nodes;
        vector<vector<int>> net;
        vector<vector<int>> edges;
        FeatureDistribution* feat_dist;
        vector<int> perc_results;
        vector<double> small_comp_results;
};