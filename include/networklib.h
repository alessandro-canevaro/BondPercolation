using namespace std;

class Network{
    public:
        Network(int n);

        void getUniformDegreeSequence(int a, int b);

        void matchStubs();

        void printNetwork();


    protected:
        int nodes;
        vector<int> sequence;
        vector<vector<int>> network;
};