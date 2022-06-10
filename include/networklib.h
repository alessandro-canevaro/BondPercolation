using namespace std;

class Network{
    public:
        Network(int n);

        void generateUniformDegreeSequence(int a, int b);

        void generateBinomialDegreeSequence(int n, float p);

        void generatePowerLawDegreeSequence(float alpha);

        void matchStubs();

        void printNetwork();

        void nodePercolation();

        vector<int> getSr();

    protected:
        int nodes;
        vector<int> sequence;
        vector<vector<int>> network;

        vector<int> sr;
};

class GiantCompSize{
    public:
        GiantCompSize(); //prepare the nets

        void generateNetworks(int net_num, int net_size, char type, float param1, float param2);

        vector<double> computeAverageGiantClusterSize(int bins); //average sr and 15.43

    protected:
        vector<vector<int>> sr_mat;

        vector<double> getBinomialPMF(float phi, int nodes);
        
        double computeGiantClusterSize(float phi, vector<double> sr);

        vector<vector<int>> transpose(vector<vector<int>> data);

        double average(vector<int> data);
};