/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

class Percolation{

    public:
        Percolation(Network* network);

        vector<int> UniformNodeRemoval();

        vector<int> HighestDegreeNodeRemoval();

        vector<int> UniformEdgeRemoval();

        vector<int> FeatureEdgeRemoval(int mu);

        vector<int> CorrFeatureEdgeRemoval();

        vector<int> TemporalFeatureEdgeRemoval();

    protected:

        void nodePercolation(vector<int> node_order);
        void edgePercolation(vector<vector<int>> edge_order);

        random_device rd;
        mt19937 rand_gen;
        int nodes;
        Network *net;
        FeatureDistribution *feat_dist;
        vector<int> perc_results;
};