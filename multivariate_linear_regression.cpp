#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> feature_mean;
vector<double> feature_std;
double y_mean, y_std;

double dot_product(const vector<double> &a, const vector<double> &b){
    double s = 0;
    for(int i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

void zscore_normalize_X(vector<vector<double>> &X){
    int m = X.size();
    int n = X[0].size();

    feature_mean.resize(n);
    feature_std.resize(n);

    for(int j = 0; j < n; j++){
        double mean = 0.0;
        for(int i = 0; i < m; i++)
            mean += X[i][j];
        mean /= m;
        feature_mean[j] = mean;

        double variance = 0.0;
        for(int i = 0; i < m; i++)
            variance += (X[i][j] - mean)*(X[i][j] - mean);
        variance /= m;
        feature_std[j] = sqrt(variance);

        for(int i = 0; i < m; i++)
            X[i][j] = (X[i][j] - feature_mean[j]) / feature_std[j];
    }
}

void zscore_normalize_y(vector<double> &y){
    int m = y.size();

    double mean = 0.0;
    for(double v : y) mean += v;
    mean /= m;
    y_mean = mean;

    double variance = 0.0;
    for(double v : y)
        variance += (v - y_mean)*(v - y_mean);
    variance /= m;
    y_std = sqrt(variance);

    for(double &v : y)
        v = (v - y_mean) / y_std;
}

vector<double> normalize_input(const vector<double> &x){
    vector<double> nx(x.size());
    for(int j = 0; j < x.size(); j++)
        nx[j] = (x[j] - feature_mean[j]) / feature_std[j];
    return nx;
}

double compute_cost(vector<vector<double>> &X, vector<double> &y,
                    vector<double> &W, double b){
    int m = X.size();
    double cost = 0.0;

    for(int i = 0; i < m; i++){
        double prediction = dot_product(X[i], W) + b;
        double error = prediction - y[i];
        cost += error*error;
    }
    return cost / (2.0*m);
}

void compute_gradient(vector<vector<double>> &X, vector<double> &y,
                      vector<double> &W, double b,
                      vector<double> &dj_dw, double &dj_db){
    int n = X[0].size();
    int m = X.size();

    for(int j = 0; j < n; j++) dj_dw[j] = 0.0;
    dj_db = 0.0;

    for(int i = 0; i < m; i++){
        double error = dot_product(X[i], W) + b - y[i];
        for(int j = 0; j < n; j++)
            dj_dw[j] += error * X[i][j];
        dj_db += error;
    }

    for(int j = 0; j < n; j++)
        dj_dw[j] /= m;
    dj_db /= m;
}

void gradient_descent(vector<vector<double>> &X, vector<double> &y,
                      vector<double> &W, double b, int epochs, double lr){
    vector<double> dj_dw(W.size());
    double dj_db = 0.0;

    for(int i = 0; i < epochs; i++){
        compute_gradient(X, y, W, b, dj_dw, dj_db);

        for(int j = 0; j < W.size(); j++)
            W[j] -= lr * dj_dw[j];
        b -= lr * dj_db;

        if(i % 100 == 0){
            cout << "Epoch: " << i
                 << " Cost: " << compute_cost(X, y, W, b)
                 << " W: ";
            for(double w : W) cout << w << " ";
            cout << "b: " << b << endl;
        }
    }
}

double predict_from_normalized(const vector<double> &x_norm,
                               const vector<double> &W, double b){
    double y_norm = dot_product(x_norm, W) + b;
    return y_norm * y_std + y_mean;
}

double predict_from_raw(const vector<double> &x_raw,
                        const vector<double> &W, double b){
    vector<double> nx = normalize_input(x_raw);
    double y_norm = dot_product(nx, W) + b;
    return y_norm * y_std + y_mean;
}

double compute_r2_score(const vector<vector<double>> &X,
                        const vector<double> &y_raw,
                        const vector<double> &W, double b){
    int m = y_raw.size();
    double mean = 0.0;
    for(double v : y_raw) mean += v;
    mean /= m;

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for(int i = 0; i < m; i++){
        double pred = predict_from_normalized(X[i], W, b);
        ss_res += (y_raw[i] - pred)*(y_raw[i] - pred);
        ss_tot += (y_raw[i] - mean)*(y_raw[i] - mean);
    }
    return 1.0 - (ss_res / ss_tot);
}

int main(){
    vector<vector<double>> X = {
        {1500.0, 3.0, 1.0, 25.0}, {2200.0, 4.0, 2.0, 10.0}, { 950.0, 2.0, 1.0, 40.0},
        {3500.0, 5.0, 3.0,  5.0}, {1800.0, 3.0, 2.0, 15.0}, {1200.0, 2.0, 1.0, 30.0},
        {2800.0, 4.0, 2.0,  8.0}, {1050.0, 2.0, 1.0, 55.0}, {4000.0, 6.0, 3.0,  2.0},
        {1650.0, 3.0, 1.0, 20.0},
        {2500.0, 4.0, 2.0, 12.0}, { 850.0, 2.0, 1.0, 60.0}, {3200.0, 5.0, 2.0,  7.0},
        {1700.0, 3.0, 2.0, 18.0}, {1350.0, 3.0, 1.0, 35.0}, {2900.0, 4.0, 3.0, 10.0},
        {1100.0, 2.0, 1.0, 50.0}, {3800.0, 5.0, 3.0,  4.0}, {1450.0, 3.0, 1.0, 28.0},
        {2050.0, 4.0, 2.0, 16.0},
        { 750.0, 1.0, 1.0, 65.0}, {3100.0, 5.0, 2.0,  9.0}, {1900.0, 3.0, 2.0, 22.0},
        {1250.0, 2.0, 1.0, 45.0}, {2700.0, 4.0, 2.0, 11.0}, {1000.0, 2.0, 1.0, 58.0},
        {3600.0, 6.0, 3.0,  3.0}, {1600.0, 3.0, 1.0, 26.0}, {2400.0, 4.0, 2.0, 14.0},
        { 900.0, 2.0, 1.0, 52.0},
        {3300.0, 5.0, 2.0,  6.0}, {1750.0, 3.0, 2.0, 17.0}, {1300.0, 2.0, 1.0, 38.0},
        {3000.0, 4.0, 3.0,  9.0}, {1150.0, 2.0, 1.0, 48.0}, {3700.0, 5.0, 3.0,  5.0},
        {1400.0, 3.0, 1.0, 32.0}, {2100.0, 4.0, 2.0, 13.0}, { 800.0, 1.0, 1.0, 62.0},
        {3400.0, 5.0, 2.0,  8.0},
        {1850.0, 3.0, 2.0, 21.0}, {1550.0, 3.0, 1.0, 23.0}, {2300.0, 4.0, 2.0, 19.0},
        {1075.0, 2.0, 1.0, 42.0}, {3900.0, 6.0, 3.0,  1.0}, {1675.0, 3.0, 1.0, 27.0},
        {2600.0, 4.0, 2.0, 15.0}, { 975.0, 2.0, 1.0, 53.0}, {3050.0, 5.0, 2.0, 10.0},
        {1525.0, 3.0, 2.0, 24.0}
    };

    vector<double> y = {
        280.0, 450.0, 175.0, 720.0, 360.0, 220.0, 550.0, 190.0, 900.0, 310.0,
        480.0, 150.0, 650.0, 330.0, 250.0, 580.0, 200.0, 820.0, 270.0, 410.0,
        130.0, 620.0, 375.0, 230.0, 510.0, 185.0, 750.0, 300.0, 460.0, 165.0,
        680.0, 345.0, 240.0, 600.0, 210.0, 780.0, 260.0, 430.0, 140.0, 700.0,
        390.0, 290.0, 440.0, 195.0, 850.0, 320.0, 495.0, 170.0, 610.0, 355.0
    };

    vector<double> y_raw = y;

    zscore_normalize_X(X);
    zscore_normalize_y(y);

    vector<double> W(4, 0.0);
    double b = 0.0;

    gradient_descent(X, y, W, b, 2000, 0.01);

    cout << "\nFinal Params:\nw: ";
    for(double w : W) cout << w << " ";
    cout << "\nb: " << b << endl;

    cout << "\nPrediction vs Actual\n";
    for(int i = 0; i < X.size(); i++){
        double pred = predict_from_normalized(X[i], W, b);
        cout << "Actual: " << y_raw[i] << " Prediction: " << pred << endl;
    }

    cout << "\nR2 Score: " << compute_r2_score(X, y_raw, W, b) << endl;

    return 0;
}