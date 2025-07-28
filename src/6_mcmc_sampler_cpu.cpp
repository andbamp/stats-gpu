// File: mcmc_sampler_cpu.cpp
#include <iostream>
#include <vector>
#include <random>
#include <cmath>   // For log, exp, pow, sqrt
#include <numeric> // For std::accumulate
#include <chrono>  // For seeding the random number generator

// --- Helper function to compute the log posterior on the CPU ---
// This is the direct equivalent of the __device__ function.
float log_posterior(float mu, const std::vector<float> &data, float sigma2, float mu0, float tau2_0)
{
    // 1. Calculate log prior: log( N(mu | mu0, tau2_0) )
    float log_prior = -0.5f * std::pow(mu - mu0, 2) / tau2_0;

    // 2. Calculate log likelihood: log( N(data | mu, sigma2) )
    float log_likelihood = 0.0f;
    for (float val : data)
    {
        log_likelihood += -0.5f * std::pow(val - mu, 2) / sigma2;
    }

    return log_prior + log_likelihood;
}

// --- The MCMC simulation function for the CPU ---
// This function replaces the CUDA kernel. It loops through each chain serially.
void run_mcmc_cpu(std::vector<float> &output, const std::vector<float> &data,
                  int N_chains, int N_iters, int N_burn_in,
                  float sigma2, float mu0, float tau2_0, float prop_sigma)
{

    // Setup a high-quality random number generator for the CPU
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<float> prop_dist(0.0f, prop_sigma);
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    // Loop over each chain (this was done in parallel on the GPU)
    for (int chain_idx = 0; chain_idx < N_chains; ++chain_idx)
    {
        float current_mu = mu0; // Start each chain at the prior mean

        // Main MCMC loop for this specific chain
        for (int i = 0; i < N_iters + N_burn_in; ++i)
        {
            // Propose a new mu using the normal distribution
            float proposed_mu = current_mu + prop_dist(generator);

            // Calculate log posterior for current and proposed mu
            float log_post_current = log_posterior(current_mu, data, sigma2, mu0, tau2_0);
            float log_post_proposed = log_posterior(proposed_mu, data, sigma2, mu0, tau2_0);

            // Acceptance check in log-space
            float log_alpha = std::min(0.0f, log_post_proposed - log_post_current);
            if (std::log(uniform_dist(generator)) < log_alpha)
            {
                current_mu = proposed_mu;
            }
        }

        // After burn-in and iterations, store the final sample of the chain
        output[chain_idx] = current_mu;
    }
}

// --- The Main Host Function ---
int main()
{
    // --- 1. Problem Setup ---
    int N_data = 1000;
    int N_chains = 1024 * 16; // Number of MCMC chains to run (16,384)
    int N_iters = 2000;
    int N_burn_in = 500;

    float true_mu = 10.0f;
    float sigma2 = 4.0f;

    float mu0 = 0.0f;
    float tau2_0 = 100.0f;

    float prop_sigma = 1.0f;

    // --- 2. Host-side Data Generation and Memory Allocation ---
    // Using std::vector for automatic memory management
    std::vector<float> h_data(N_data);
    std::vector<float> h_output(N_chains);

    // Generate synthetic data using C++'s <random> library
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<float> data_dist(true_mu, std::sqrt(sigma2));

    for (int i = 0; i < N_data; ++i)
    {
        h_data[i] = data_dist(generator);
    }

    // --- 5. Launch the MCMC Simulation on the CPU ---
    printf("Launching %d serial MCMC chains on the CPU...\n", N_chains);
    run_mcmc_cpu(h_output, h_data, N_chains, N_iters, N_burn_in,
                 sigma2, mu0, tau2_0, prop_sigma);
    printf("MCMC simulation finished.\n");

    // --- 6. Analyze results ---
    // Use std::accumulate for a clean and efficient sum
    float posterior_mean = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    posterior_mean /= N_chains;

    printf("\n--- Analysis ---\n");
    printf("Posterior Mean of mu (from %d samples): %f\n", N_chains, posterior_mean);
    printf("True Mean of mu was: %f\n", true_mu);

    // --- 7. Cleanup ---
    // No manual free/delete needed thanks to std::vector!

    return 0;
}
