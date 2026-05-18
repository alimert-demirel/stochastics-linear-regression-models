#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

const double S0 = 40.0;           // Approximate current price
const double MU = 0.08;          // 8% expected annual return
const double SIGMA = 0.25;       // 25% annual volatility
const double T = 1.0;           // 1 Year
const int STEPS = 252;            // Trading days
const int PATHS = 1000;          // Number of Monte Carlo smulations

const double JUMP_INTENSITY = 2.0;   // Average 2 crashes/shocks per year
const double JUMP_MEAN = -0.15;     // Average drop is 15%
const double JUMP_VOL = 0.05;        // Jump volatilty

int main() {
    cout << "Initializing High-Frequency C++ Monte Carlo Engine..." << endl;
    
    double dt = T / STEPS;
    mt19937 generator(42); // Seeded for reproducibility.
    normal_distribution<double> standard_normal(0.0, 1.0);
    poisson_distribution<int> poisson(JUMP_INTENSITY * dt);
    normal_distribution<double> jump_normal(JUMP_MEAN, JUMP_VOL);

    ofstream gbm_file("../../data/gbm_paths.csv");
    ofstream merton_file("../../data/merton_paths.csv");

    for (int p = 0; p < PATHS; ++p) {
        double current_gbm = S0;
        double current_merton = S0;
        
        gbm_file << current_gbm;
        merton_file << current_merton;

        for (int t = 1; t <= STEPS; ++t) {
            double z = standard_normal(generator);
            
            double drift = (MU - 0.5 * SIGMA * SIGMA) * dt;
            double diffusion = SIGMA * sqrt(dt) * z;
            current_gbm = current_gbm * exp(drift + diffusion);
            
            int num_jumps = poisson(generator);
            double jump_multiplier = 1.0;
            for (int j = 0; j < num_jumps; ++j) {
                jump_multiplier *= exp(jump_normal(generator));
            }
            current_merton = current_merton * exp(drift + diffusion) * jump_multiplier;

            gbm_file << "," << current_gbm;
            merton_file << "," << current_merton;
        }
        gbm_file << "\n";
        merton_file << "\n";
    }

    gbm_file.close();
    merton_file.close();
    cout << "Simulated 1,000 paths for both GBM and Merton." << endl;
    cout << "Data exported to 'data/gbm_paths.csv' and 'data/merton_paths.csv'." << endl;
    
    return 0;
}