/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

class Percolation{

    public:
        Percolation(vector<vector<int>> network);

        vector<int> UniformNodeRemoval();

        vector<int> HighestDegreeNodeRemoval();

        vector<int> UniformEdgeRemoval();

        vector<int> FeatureEdgeRemoval();

        vector<int> CorrFeatureEdgeRemoval();

        vector<int> TemporalFeatureEdgeRemoval();

    protected:

        void nodePercolation();
        void edgePercolation();

        random_device rd;
        mt19937 rand_gen;
        int nodes;
        vector<vector<int>> network;
        vector<vector<int>> edge_list;
};