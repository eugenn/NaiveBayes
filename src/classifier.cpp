#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {
    features_cnt = 0;

    for (int i = 0; i < 4; i++) {
        mean_left[i] = 0.0;
        mean_right[i] = 0.0;
        mean_keep[i] = 0.0;

        var_left[i] = 0.0;
        var_right[i] = 0.0;
        var_keep[i] = 0.0;

        sum_left[i] = 0.0;
        sum_right[i] = 0.0;
        sum_keep[i] = 0.0;
    }
}

GNB::~GNB() {


}

double gaussian_prob(double &obs, double &mu, double &sig) {
    const double num = pow((obs - mu), 2.0);
    const double denum = 2 * pow(sig, 2);
    const double norm = 1 / sqrt(2 * M_PI * pow(sig, 2));

    return norm * exp(-num / denum);
}

void GNB::train(vector<vector<double>> data, vector<string> labels) {

    int cnt_left = 0;
    int cnt_keep = 0;
    int cnt_right = 0;

    features_cnt = data[0].size();

    //Calculate the sum
    for (int i = 0; i < data.size(); i++) {

        for (int var = 0; var < features_cnt; var++) {

            if (labels[i] == "left") {
                sum_left[var] += data[i][var];
                cnt_left++;
            } else if (labels[i] == "right") {
                sum_right[var] += data[i][var];
                cnt_right++;
            } else if (labels[i] == "keep") {
                sum_keep[var] += data[i][var];
                cnt_keep++;
            }

        }
    }

    cnt_left = cnt_left / 4;
    cnt_right = cnt_right / 4;
    cnt_keep = cnt_keep / 4;

    cout << "cnt_left: " << cnt_left << endl;
    cout << "cnt_keep: " << cnt_keep << endl;
    cout << "cnt_right: " << cnt_right << endl;

    float total_cnt = cnt_left + cnt_right + cnt_keep;

    cout << "total count = " << total_cnt << endl;

    //Calculate the mean
    for (int var = 0; var < features_cnt; var++) {
        mean_left[var] = sum_left[var] / cnt_left;
        mean_keep[var] = sum_keep[var] / cnt_keep;
        mean_right[var] = sum_right[var] / cnt_right;
    }

    //print mean
    for (int var = 0; var < features_cnt; var++) {
        cout << "mean_left[" << var << "]" << mean_left[var] << endl;
        cout << "mean_keep[" << var << "]" << mean_keep[var] << endl;
        cout << "mean_right[" << var << "]" << mean_right[var] << endl;
    }

    //Calculate the variance
    for (int i = 0; i < data.size(); i++) {
        for (int var = 0; var < features_cnt; var++) {
            if (labels[i] == "left") {
                var_left[var] += pow((data[i][var] - mean_left[var]), 2);
            } else if (labels[i] == "right") {
                var_right[var] += pow((data[i][var] - mean_right[var]), 2);
            } else if (labels[i] == "keep") {
                var_keep[var] += pow((data[i][var] - mean_keep[var]), 2);
            }

        }

    }

    for (int var = 0; var < features_cnt; var++) {
        var_left[var] = sqrt(var_left[var] / cnt_left);
        var_keep[var] = sqrt(var_keep[var] / cnt_keep);
        var_right[var] = sqrt(var_right[var] / cnt_right);
    }

    //print mean and variance
    for (int var = 0; var < features_cnt; var++) {
        cout << "var_left[" << var << "]" << var_left[var] << endl;
        cout << "var_keep[" << var << "]" << var_keep[var] << endl;
        cout << "var_right[" << var << "]" << var_right[var] << endl;
    }

    for (int j = 0; j < features_cnt; j++) {
        means[0][j] = mean_left[j];
        means[1][j] = mean_keep[j];
        means[2][j] = mean_right[j];

        vars[0][j] = var_left[j];
        vars[1][j] = var_keep[j];
        vars[2][j] = var_right[j];
    }
}

string GNB::predict(vector<double> sample) {
    vector<double> probs;

    for (int k = 0; k < possible_labels.size(); k++) {
        double product = 1.0;
        for (int i = 0; i < sample.size(); i++) {
            double obs = sample[i];

            double mu = means[k][i];
            double sig = vars[k][i];
            double o = obs;

            double likehood = gaussian_prob(o, mu, sig);
            cout << "o=" << o << " mu=" << mu << " sig=" << sig << endl;
            product *= likehood;
            cout << "likehood=" << likehood << endl;
        }

        probs.push_back(product);
    }

    double sum_of_elems = 0;

    std::for_each(probs.begin(), probs.end(), [&](double &n) {
        sum_of_elems += n;
    });

    std::transform(probs.begin(), probs.end(), probs.begin(),
                   [sum_of_elems](double &p) -> double { return p / sum_of_elems; });


    int index = 0;
    double best_p = 0;

    for (int j = 0; j < probs.size(); j++) {
        double p = probs[j];

        if (p > best_p) {
            best_p = p;
            index = j;
        }
    }

    return this->possible_labels[index];

}