/**
* Classes for generating a network, perform percolation and compute properties
* @file networklib.h
* @author Alessandro Canevaro
* @version 11/06/2022
*/

using namespace std;

class DegreeDistribution{

    public:
        DegreeDistribution(int net_size);

        void generateBinomialDD(float p);

        void generateFixedDD(int k);

        void generateGeometricDD(float p);

        void generatePowerLawDD(float alpha);

        vector<int> getDegreeDistribution();

    protected:

        int nodes;
        vector<int> degree_dist;
};