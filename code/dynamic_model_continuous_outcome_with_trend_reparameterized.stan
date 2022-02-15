data {
  int<lower=0> includeTrend; // Set includeTrend = 0 if trends should not be modeled, any other value will allow for trends in the parameters.

  int<lower=0> N; // # observations
  int<lower=0> p;   // number of predictors
  int<lower=0> timePeriods; // # of time periods
  int<lower=0> futureTimePeriods; // # of future time periods for which we want predictions
  // array[timePeriods] int s; // # observations in each time period
  int s[timePeriods]; // the same before Stan version 2.27

  // prior mean and covariance matrix of the state vector (taken from https://github.com/stan-dev/example-models/tree/master/misc/dlm)
  // assume for simplicity that this is identical for all covariates?
  vector[p] m0;
  matrix[p, p] C0;
  // and for the trend vector
  vector[p] m0_nu;
  matrix[p, p] C0_nu;
  
  // data
  matrix[N, p] x;   // predictor matrix
  real y[N];
  // vector[N] y;		// outcome vector
}
parameters {
  real<lower = 0> sigma_alpha[p];
  real<lower = 0> sigma_betaObs[p];
  real<lower = 0> sigma_nu[p];
  real<lower = 0> sigma_y;
  matrix[timePeriods + futureTimePeriods, p] betaObs_raw;
  matrix[timePeriods + futureTimePeriods, p] alpha;
  matrix[timePeriods + futureTimePeriods, p] nu;
}
transformed parameters {
  // for reasons for reparameterization, see https://mc-stan.org/docs/2_28/stan-users-guide/reparameterization.html (Neals Funnel) and https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
  matrix[timePeriods + futureTimePeriods, p] betaObs;
  // implies betaObs[:, varcounter] ~ normal(alpha[1:timePeriods,varcounter], sigma_betaObs[varcounter]);
  for (varcounter in 1:p) {
    betaObs[:, varcounter] = alpha[:, varcounter] + sigma_betaObs[varcounter] * betaObs_raw[:, varcounter];
  }
}
model {
  int pos;

  // not exactly following this recommendation: https://mc-stan.org/docs/2_28/stan-users-guide/hierarchical-priors.html#ref-ChungEtAl:2013, which suggest 1 instead of 1.05
  sigma_alpha ~ gamma(rep_vector(1.05, p), rep_vector(1.0/10.0, p));
  sigma_betaObs ~ gamma(rep_vector(1.05, p), rep_vector(1.0/10.0, p));
  sigma_y ~ gamma(1.05, 1.0/10.0);

  // trend modeling
  sigma_nu ~ gamma(rep_vector(1.05, p), rep_vector(1.0/10.0, p));

  // loop over all covariates, can this be parallelized?
  for (varcounter in 1:p) {
    if (includeTrend) {
      // independence assumption between covariates encoded here
      nu[1, varcounter] ~ normal(m0_nu[varcounter], C0_nu[varcounter,varcounter]);
      nu[2:(timePeriods + futureTimePeriods), varcounter] ~ normal(nu[1:(timePeriods + futureTimePeriods - 1), varcounter], sigma_nu[varcounter]);
    }

    // independence assumption between covariates encoded here
    alpha[1, varcounter] ~ normal(m0[varcounter], C0[varcounter,varcounter]);
    // use nu only if includeTrend != 0
    alpha[2:(timePeriods + futureTimePeriods), varcounter] ~ normal(includeTrend ? alpha[1:(timePeriods + futureTimePeriods - 1),varcounter] + nu[1:(timePeriods + futureTimePeriods - 1),varcounter] : alpha[1:(timePeriods + futureTimePeriods - 1), varcounter], sigma_alpha[varcounter]);

    // fluctuation equation (because we have multiple observations for each point in time)
    // betaObs[:, varcounter] ~ normal(alpha[1:timePeriods,varcounter], sigma_betaObs[varcounter]);
    betaObs_raw[:, varcounter] ~ std_normal();
  }

  // observation equation
  pos = 1;
  for (t in 1:timePeriods) {
    // segment and block use only observations from time period t
	  segment(y, pos, s[t]) ~ normal(to_vector(block(x, pos, 1, s[t], p) * to_vector(betaObs[t,:])), sigma_y);
	  pos = pos + s[t];
  }
}
