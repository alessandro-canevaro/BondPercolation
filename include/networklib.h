/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

/**
* Main class for generating the network according to the configurational model
* and performing node percolation
*/
class Network{

    public:

        /** Default constructor
        * @param n number of nodes in the network
        */
        Network(int n);

        Network(string path);

        void generateFixedDegreeSequence(int k);

        /** Generates a degree sequence according to the uniform distribution
        * @param a lower limit of the uniform distribution
        * @param b upper limit of the uniform distribution
        */
        void generateUniformDegreeSequence(int a, int b);

        void generateGeometricDegreeSequence(float p);

        /** Generates a degree sequence according to the binomial distribution
        * @param n number of trials
        * @param p success probability
        */
        void generateBinomialDegreeSequence(int n, float p);

        /** Generates a degree sequence according to the power-law distribution
        * @param alpha exponent
        * @param x_max upper limit of the uniform distribution (lower is 1)
        */
        void generatePowerLawDegreeSequence(float alpha);

        /** Computes the average value of the previously generated degree sequence
        * @return mean value of the degree sequence
        */
        float getDegreeDistMean();

        /** Add edges to the network by matching stubs.
        * the neber of stubs for each node depends on the previously generate degree sequence.
        */
        void matchStubs();

        /** Removes self and multi edges from the network.
        */
        void removeSelfMultiEdges();

        /** Print on the console the network.
        */
        void printNetwork();

        /** Perform node percolation.
        * at each added node, it saves in the vetor sr the size of the largest cluster
        */
        void nodePercolation(bool small_comp = false);

        /** return the network as a vector of vectors containing the nodes and edges
        * @return network
        */
        vector<vector<int>> getNetwork();

        /** return the percolation results, namely the vector containing the size of the largest cluster at each stage.
        * @return sr
        */
        vector<int> getSr();

        vector<double> getSs();

        void generateUniformOrder();

        bool isConnected();

        int getGiantClusterSize();

        vector<double> getNeighborDegreeAvg();  

        float getSecondOrderMoment();

        void generateHighestDegreeFirstOrder();

        vector<int> getSasfunctionofK(int max_degree);

        void linkPercolation();

        vector<int> getSe();

        void generateUniformEdgeOrder();

        void equalizeEdgeNumber(int M);

        void generateFeatureEdgeOrder();

        void generateCorrFeatureEdgeOrder();

        vector<int> getSasfunctionofF(int max_f);

    protected:

        int nodes;
        vector<int> sequence;
        vector<vector<int>> network;
        vector<int> node_order;
        vector<int> sr;
        vector<int> se;
        vector<double> ss; //small comp.
        vector<vector<int>> edge_order;
        vector<int> feature_values;
};

/**
* Class for computing the giant component size as a function of the occupation probability
*/
class GiantCompSize{

    public:

        /** Default constructor
        */
        GiantCompSize();

        /** Generates the networks and run node percolation
        * @param net_num number of networks
        * @param net_size number of nodes in the network
        * @param type method to be used for generating the degree sequence (u=uniform, b=binomial, p=power-law)
        * @param param1 first parameter passed to the degree sequence generating function
        * @param param2 second parameter passed to the degree sequence generating function
        */
        void generateNetworks(int net_num, int net_size, char type, char percolation_type, float param1, float param2);

        /** Computes the average size of the giant component for a range of values of the occupation probability
        * @param bins the occupation probability range is splitted into bins values
        * @return vector of size bins containing the results for the diferrent values of the occupation probability
        */
        vector<double> computeAverageGiantClusterSize(int bins);

        vector<double> computeBinomialAverage(int bins);

        vector<double> computeAverageGiantClusterSizeAsFunctionOfK();

        void loadBinomialPMF(string path);

        void printNeighborDegreeAvg(int net_num, int net_size, char type, float param1, float param2);

    protected:

        /** Computes the average size of the giant component for a given value of the occupation probability
        * @param phi occupation probability
        * @param sr vector containing the result of the percolation process
        * @return average size of the giant component
        */
        double computeGiantClusterSize(float phi, vector<double> sr);

        /** Computes the binomial probability mass function
        * @param phi occupation probability
        * @param nodes number of nodes in the network
        * @return vector of binomial pmf for all the values of k in range (1, nodes)
        */
        vector<double> getBinomialPMF(float phi, int nodes);

        vector<vector<double>> getBinomialPMF(string path);

        /** Transpose a vector of vector matrix
        * @param data input matrix
        * @return transposed input matrix
        */
        vector<vector<int>> transpose(vector<vector<int>> data);

        vector<vector<double>> transpose(vector<vector<double>> data);

        /** Computes the average of a vector
        * @param data input vector
        * @return average value of the input vector
        */
        double average(vector<int> data, bool all=true);

        double average(vector<double> data);

        vector<vector<int>> sr_mat;

        vector<vector<int>> sk_mat;

        vector<vector<double>> ss_mat;

        vector<vector<double>> bin_pmf;
};