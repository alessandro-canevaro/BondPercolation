/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

class Network{

    public:

        Network(vector<int> degree_sequence);

        Network(vector<vector<int>> edge_list);

        void equalizeEdges(int m);

        void printNetwork();

        vector<vector<int>> getEdgeList();

        vector<vector<int>> getNetwork();

    protected:

        void matchStubs(vector<int> degree_sequence);

        void removeSelfMultiEdges();

        void makeEdgeList();

        int nodes;
        vector<vector<int>> network;
        vector<vector<int>> edge_list;
};