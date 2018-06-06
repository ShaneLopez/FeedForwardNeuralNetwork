#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cmath>

using namespace std;

class network
{
private:
    double lr;
    vector < double > inputs;
    vector < double > hidden;
    vector < double > outputs;
    vector < vector < double > > ihweights;
    vector < vector < double > > howeights;
public:
    void init (double ulr, int inputnum, int hiddennum, int outputnum);
    void forwardpass (vector <double> uinput);
    void backprop (vector<double> error);
    void getoutputs(vector<double>& uoutput);
};
