using namespace std;

class Network{
    public:
        Network(int n);

        void getUniformDegreeSequence(int a, int b);

        void getBinomialDegreeSequence(int n, float p);

        void matchStubs();

        void printNetwork();

        vector<int> nodePercolation();

        float computeMeanSizeOfLargestCluster(float phi, vector<int> sr);

        vector<float> computeGiantClusterPlot(int bins, vector<int> sr);


    protected:
        int nodes;
        vector<int> sequence;
        vector<vector<int>> network;
};

long long int binomialCoeff(const int n, const int k);