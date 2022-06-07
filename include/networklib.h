using namespace std;

class Network{
    public:
        Network(int n);

        void getUniformDegreeSequence(int a, int b);

        void getBinomialDegreeSequence(int n, float p);

        void matchStubs();

        void printNetwork();

        void nodePercolation();


    protected:
        int nodes;
        vector<int> sequence;
        vector<vector<int>> network;
};