#include <iostream>
#include <vector>

using namespace std;

double f_wb(double w, double b, double x_i) {
    return (w * x_i + b);
}

double compute_cost(double w, double b, vector<int> &x, vector<int> &y) {
    int m = x.size();
    double cost = 0;

    for (int i = 0; i < m; ++i)
        cost += ((f_wb(w, b, x[i]) - y[i]) * (f_wb(w, b, x[i]) - y[i]));

    cost /= (2 * m);
    return cost;
}

double compute_r2_score(double w, double b, vector<int> &x, vector<int> &y){
    int m = x.size();
    double ss_tot = 0.0, ss_res = 0.0;
    double mean_y = 0.0;
    for (int i = 0; i < m; ++i) mean_y += y[i];
    mean_y /= m;

    for (int i = 0; i < m; ++i) {
        double y_pred = w * x[i] + b;
        ss_tot += ((y[i] - mean_y) * (y[i] - mean_y));
        ss_res += ((y[i] - y_pred) * (y[i] - y_pred));
    }

    return 1 - (ss_res / ss_tot);
}

void compute_gradient(double w, double b, vector<int> &x, vector<int> &y, double &dj_dw, double &dj_db) {
    int m = x.size();
    dj_dw = 0;
    dj_db = 0;

    for (int i = 0; i < m; ++i) {
        dj_dw += ((f_wb(w, b, x[i]) - y[i]) * x[i]);
        dj_db += (f_wb(w, b, x[i]) - y[i]);
    }

    dj_dw /= m;
    dj_db /= m;
}

void gradient_descent(double &w, double &b, vector<int> &x, vector<int> &y, int epochs, double learning_rate) {
    double dj_dw, dj_db;

    for (int i = 0; i < epochs; ++i) {
        compute_gradient(w, b, x, y, dj_dw, dj_db);
        w = w - learning_rate * dj_dw;
        b = b - learning_rate * dj_db;

        if (i % 100 == 0)
            cout << "Iterations: " << i
                 << " Cost: " << compute_cost(w, b, x, y)
                 << " w: " << w << " b: " << b << endl;
    }
}

int main() {
    vector<int> x = {
        60, 110, 135, 75, 180, 55, 125, 90, 150, 85,
        105, 140, 70, 190, 50, 115, 95, 160, 80, 120,
        145, 65, 170, 45, 100, 130, 58, 112, 138, 72,
        185, 52, 122, 92, 155, 82, 108, 142, 68, 195,
        48, 118, 98, 165, 88, 128, 148, 62, 175, 42
    };

    vector<int> y = {
        185, 312, 395, 225, 518, 171, 361, 270, 425, 255,
        302, 415, 205, 550, 155, 335, 285, 458, 240, 350,
        430, 195, 490, 140, 290, 378, 180, 325, 405, 215,
        530, 165, 355, 275, 445, 248, 318, 420, 200, 570,
        150, 340, 295, 475, 265, 370, 440, 190, 505, 135
    };

    double w = 0, b = 0;
    gradient_descent(w, b, x, y, 10000, 0.0001);

    cout << "\nFinal Params:\n w: " << w << "\n b: " << b << endl;

    cout<<"Comparison: Prediction vs Actual Price\n";
    for(int i = 0; i < x.size(); ++i)
        cout<<"Size: "<< x[i]<<" Price: "<< y[i]<<" Prediction: "<< f_wb(w,b,x[i])<< endl;
    cout << "R2 Score: " << compute_r2_score(w,b,x,y) << endl;

    return 0;
}