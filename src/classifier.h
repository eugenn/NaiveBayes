#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

    vector<string> possible_labels = {"left", "keep", "right"};


    /**
      * Constructor
      */
    GNB();

    /**
     * Destructor
     */
    virtual ~GNB();

    void train(vector<vector<double> > data, vector<string> labels);

    string predict(vector<double>);

    int features_cnt;

    double mean_left[4];
    double mean_right[4];
    double mean_keep[4];
    double var_left[4];
    double var_right[4];
    double var_keep[4];
    double sum_left[4];
    double sum_right[4];
    double sum_keep[4];

    double means[3][4];
    double vars[3][4];
};

#endif



