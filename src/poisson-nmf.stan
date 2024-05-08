data {
  int<lower=1> J;
  int<lower=1> I;    
  int<lower=1> K;
  array[I, J] int<lower=0> X;
  real<lower=0> alpha;
  real<lower=0> gamma0;  // shape parameters
  real<lower=0> gamma1;  // scale parameters
  real<lower=0> delta0;  // shape parameters for mean loadings
  real<lower=0> delta1;  // scale parameters for mean loadings
}
        
transformed data {
  vector<lower=0>[I] alpha_array = rep_vector(alpha, I);
}

parameters {
  vector<lower=0>[K] nu;
  vector<lower=0>[K] mu;
  matrix<lower=0>[K, J] theta;
  array[K] simplex[I] r;
}
        
model {
  real mutation_rate;
  for (k in 1:K) {
    nu[k] ~ inv_gamma(gamma0, gamma1);
    mu[k] ~ inv_gamma(delta0, delta1);
    r[k] ~ dirichlet(alpha_array);
  }
  for (j in 1:J) {
    for (k in 1:K) {
      theta[k, j] ~ gamma(nu[k], nu[k]/mu[k]);
    }
  }

  for (j in 1:J){
    for (i in 1:I) {
      mutation_rate = 0;
      for (k in 1:K){
          mutation_rate += r[k, i] * theta[k, j];
      }
      target += -mutation_rate + X[i, j]*log(mutation_rate);
    }
  }
}
