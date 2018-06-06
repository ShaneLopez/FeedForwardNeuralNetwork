#include <windows.h>
#include "network.h"

int main()
{
    network net;
    vector< vector<double> > inputs{{-1.0,-1.0,1.0}, {-1.0,1.0,1.0}, {1.0,-1.0,1.0}, {1.0,1.0,1.0}};
    vector<double> expoutputs{1.0,1.0,-1.0,-1.0};
    vector<double> outputs;
    vector<double> error;
    double errorsum = 0;
    double earlystop = 1;
    int check = 0;

    net.init(0.1, 3, 4, 1);
    int choice;
    while(earlystop > 0.00005)
    {
        choice = rand() % inputs.size();
        net.forwardpass(inputs[choice]);
        net.getoutputs(outputs);
        for(int j = 0; j < outputs.size(); j++)
        {
            error.push_back(outputs[j] - expoutputs[choice]);
            errorsum += error[j];
        }
        check++;
        net.backprop(error);
        error.clear();

        if(check == 50)
        {
            earlystop = pow(errorsum, 2) / check;
            errorsum = 0;
            check = 0;
        }
    }

    net.forwardpass(inputs[0]);
    net.getoutputs(outputs);
    for(int i = 0; i < outputs.size(); i++)
    {
        cout << outputs[i] << endl;
    }

    net.forwardpass(inputs[1]);
    net.getoutputs(outputs);
    for(int i = 0; i < outputs.size(); i++)
    {
        cout << outputs[i] << endl;
    }

    net.forwardpass(inputs[2]);
    net.getoutputs(outputs);
    for(int i = 0; i < outputs.size(); i++)
    {
        cout << outputs[i] << endl;
    }

    net.forwardpass(inputs[3]);
    net.getoutputs(outputs);
    for(int i = 0; i < outputs.size(); i++)
    {
        cout << outputs[i] << endl;
    }
    return 0;
}
