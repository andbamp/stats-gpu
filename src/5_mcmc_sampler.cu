// File: mcmc_sampler.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <math.h> // For logf, expf, powf, sqrtf

// --- Device helper function to compute the log posterior ---
// A __device__ function can only be called from the device (e.g., from a kernel).
__device__ float log_posterior(float mu, const float *d_data, int N, float sigma2, float mu0, float tau2_0)
{
    // 1. Calculate log prior: log( N(mu | mu0, tau2_0) )
    float log_prior = -0.5f * powf(mu - mu0, 2) / tau2_0;

    // 2. Calculate log likelihood: log( N(data | mu, sigma2) )
    float log_likelihood = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        log_likelihood += -0.5f * powf(d_data[i] - mu, 2) / sigma2;
    }

    return log_prior + log_likelihood;
}

// --- The MCMC Kernel ---
__global__ void mcmc_kernel(float *d_output, const float *d_data, curandState_t *states,
                            int N_data, int N_chains, int N_iters, int N_burn_in,
                            float sigma2, float mu0, float tau2_0, float prop_sigma)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_chains)
    {
        curandState_t local_rand_state = states[idx];
        float current_mu = mu0; // Start each chain at the prior mean

        // Main MCMC loop
        for (int i = 0; i < N_iters + N_burn_in; ++i)
        {
            // Propose a new mu using the Normal distribution from cuRAND
            float proposed_mu = current_mu + curand_normal(&local_rand_state) * prop_sigma;

            // Calculate log posterior for current and proposed mu
            float log_post_current = log_posterior(current_mu, d_data, N_data, sigma2, mu0, tau2_0);
            float log_post_proposed = log_posterior(proposed_mu, d_data, N_data, sigma2, mu0, tau2_0);

            // Acceptance check in log-space
            float log_alpha = fminf(0.0f, log_post_proposed - log_post_current);
            if (logf(curand_uniform(&local_rand_state)) < log_alpha)
            {
                current_mu = proposed_mu;
            }
        }

        // After burn-in and iterations, store the final sample of the chain
        d_output[idx] = current_mu;
        states[idx] = local_rand_state; // Save the updated random state
    }
}

// The setup_kernel from the previous section
__global__ void setup_kernel(curandState_t *states, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// --- The Main Host Function ---
int main()
{
    // --- 1. Problem Setup ---
    int N_data = 1000;        // Number of data points
    int N_chains = 1024 * 16; // Number of parallel MCMC chains to run (16,384)
    int N_iters = 2000;       // MCMC iterations per chain
    int N_burn_in = 500;      // Burn-in iterations to discard

    // True parameters for data generation
    float true_mu = 10.0f;
    float sigma2 = 4.0f;

    // Priors for mu ~ N(mu0, tau2_0)
    float mu0 = 0.0f;
    float tau2_0 = 100.0f;

    // MCMC proposal width
    float prop_sigma = 1.0f;

    // --- 2. Host-side Data Generation and Memory Allocation ---
    float *h_data = (float *)malloc(N_data * sizeof(float));
    float *h_output = (float *)malloc(N_chains * sizeof(float));

    // Generate synthetic data from the true model
    // Note: This host-side RNG is simple for demonstration.
    // In a real scenario, real data would be loaded.
    srand(time(NULL));
    for (int i = 0; i < N_data; ++i)
    {
        // A simple way to get a standard normal-like random number
        float u1 = rand() / (float)RAND_MAX;
        float u2 = rand() / (float)RAND_MAX;
        float rand_std_normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        h_data[i] = true_mu + sqrtf(sigma2) * rand_std_normal;
    }

    // --- 3. Device-side Memory Allocation ---
    float *d_data, *d_output;
    curandState_t *d_states;
    cudaMalloc(&d_data, N_data * sizeof(float));
    cudaMalloc(&d_output, N_chains * sizeof(float));
    cudaMalloc(&d_states, N_chains * sizeof(curandState_t));

    // --- 4. Copy data and Setup Random States ---
    cudaMemcpy(d_data, h_data, N_data * sizeof(float), cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = 256;
    int BLOCKS_PER_GRID = (N_chains + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    setup_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_states, time(NULL));

    // --- 5. Launch the MCMC Kernel ---
    printf("Launching %d parallel MCMC chains...\n", N_chains);
    mcmc_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        d_output, d_data, d_states, N_data, N_chains, N_iters, N_burn_in,
        sigma2, mu0, tau2_0, prop_sigma);
    cudaDeviceSynchronize();
    printf("MCMC simulation finished.\n");

    // --- 6. Copy results back and analyze ---
    cudaMemcpy(h_output, d_output, N_chains * sizeof(float), cudaMemcpyDeviceToHost);

    float posterior_mean = 0.0f;
    for (int i = 0; i < N_chains; ++i)
    {
        posterior_mean += h_output[i];
    }
    posterior_mean /= N_chains;

    printf("\n--- Analysis ---\n");
    printf("Posterior Mean of mu (from %d samples): %f\n", N_chains, posterior_mean);
    printf("True Mean of mu was: %f\n", true_mu);

    // --- 7. Cleanup ---
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_states);
    free(h_data);
    free(h_output);

    return 0;
}
