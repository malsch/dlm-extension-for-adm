#################

library("colorspace"); q4 <- palette.colors(palette = "Okabe-Ito")
library("rstan")

# Create Data
set.seed(3434)

n <- 100
y_ts <-
  structure(c(1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935,
              1110, 994, 1020, 960, 1180, 799, 958, 1140, 1100, 1210, 1150, 1250, 1260, 1220,
              1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969, 831,
              726, 456, 824, 702, 1120, 1100, 832, 764, 821, 768, 845, 864, 862, 698, 845,
              744, 796, 1040, 759, 781, 865, 845, 944, 984, 897, 822, 1010, 771, 676, 649,
              846, 812, 742, 801, 1040, 860, 874, 848, 890, 744, 749, 838, 1050, 918, 986,
              797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740),
            .Dim = c(1, 100))

# calculate moving average
ma <- function(x, n = 8){stats::filter(x, rep(1 / n, n), sides = 2)}
y_ts_smooth <- ma(y_ts[1, ])

# simulate data
dt <- data.frame(ts = rep(y_ts, each = 5),
                t = rep(1:100, each = 5),
                x0 = rep(1, 500),
                x1 = rnorm(500, 0, 1),
                x2 = rnorm(500, 0, 20))
dt$y <- dt$ts * dt$x0 + 30 * dt$x1 + rnorm(5 * 100, 0, 5)
dt$y_binary1 <- rbinom(500, size = rep(1, 500), prob = 1 / (1 + exp(-((dt$ts - 919.35) * dt$x0 + 30 * dt$x1))))
dt$y_binary2 <- as.integer(dt$y > dt$ts * dt$x0 + 30 * dt$x1)
dt$y_binary3 <- as.integer(dt$y > 900)

pdf("visualize_simulated_data.pdf", width = 12)
plot(x = seq(1, 100, length = 500), dt$y, xlab = "time", ylab = "y", col = q4[1])
lines(x = seq(1, 100, length = 100), y_ts_smooth, col = q4[2], lty = "solid", lwd = 2)
points(x = seq(1, 100, length = 500), dt$ts + 30 * dt$x1, col = q4[3])
points(x = seq(1, 100, length = 100), y_ts[1,], col = q4[2], lwd = 2)
# points(x = seq(1, 100, length = 500), dt$y)
abline(v = 350 / 5)
text(x = c(200, 450) / 5, y = 1400, labels = c("Training", "Test"))
legend("bottomleft", lty = 1, legend = c("time series (with moving average)", "signal = time series + 30*x1 + 0*x2", "observed values = signal + N(0, 5)"), col = q4[c(2, 3, 1)])
dev.off()

dt_train <- dt[1:(5*70),]
dt_test <- dt[(1+(5*70)):500,]

dt_test_X <- dt_test

############################################
# test continuous outcome y <- dt$ts * dt$x0 + 30 * dt$x1 + rnorm(5 * 100, 0, 5)
############################################

numCovariates <- 3
dynamicModelData = list(
    includeTrend = 0,
    N = nrow(dt_train),
    p = numCovariates,
    timePeriods = length(unique(dt_train$t)),
    futureTimePeriods = 30, # for how many time periods in the future do we need predictions?
    s = data.table::data.table(dt_train)[, .N, by = .(t)]$N,
    m0 = rep(0, numCovariates),
    C0 = 1e+07 * diag(numCovariates),
    m0_nu = rep(0, numCovariates),
    C0_nu = 1e+07 * diag(numCovariates),
    x = data.frame(dt_train$x0, dt_train$x1, dt_train$x2),
    y = dt_train$y
)

dyn_fit <- stan(file = "dynamic_model_continuous_outcome_with_trend_reparameterized.stan", data = dynamicModelData, iter=2000, chains=8, cores = 8,
#               no progress after 10 iterations with the following: control = list(max_treedepth = 15, adapt_delta = 0.9)
               control = list(adapt_delta = 0.9, stepsize = 0.8), # one might also try: algorithm = 'HMC'
                open_progress = FALSE, verbose = TRUE, refresh = 100)


la <- extract(dyn_fit, permuted = TRUE) # return a list of arrays

ndraws <- 8000
# get predicted future values for betaObs and for predicted y
futureBetaObs <- array(NA_real_, c(ndraws, 100, 3))
futurePredictedY <- array(NA_real_, c(nrow(dt_test_X), ndraws))
futurePredictedY_var_too_low <- array(NA_real_, c(nrow(dt_test_X), ndraws))
for (t in 71:100) {
    for (p in 1:3) {
        futureBetaObs[, t, p] <- rnorm(ndraws, la$alpha[, t, p], la$sigma_betaObs[, p])
    }

    for (draw in 1:ndraws) {
        if (length(which(dt_test$t == t)) > 0) { # run the following only if we have covariates at time t
            futurePredictedY[which(dt_test$t == t), draw] <- as.matrix(dt_test_X[which(dt_test$t == t), c("x0", "x1", "x2")]) %*% la$betaObs[draw, t, ]
            futurePredictedY_var_too_low[which(dt_test$t == t), draw] <- as.matrix(dt_test_X[which(dt_test$t == t), c("x0", "x1", "x2")]) %*% la$alpha[draw, t, ]
        }
    }
}

# note that la$alpha and la$betaObs are very similar, but we would underestimate variances when la$alpha gets used
apply(la$alpha, 2:3, mean) #  reflects true values over time
apply(la$betaObs, 2:3, mean) # same
apply(la$alpha, 2:3, sd)[1:10,] # but variances of MCMC draws are not identical (sd(la$alpha[,1,1]))
apply(la$betaObs, 2:3, sd)[1:10,] # but variances are not identical

## few signs that chains did not mix..
plot(futurePredictedY[49,])
plot(la$betaObs[, 30, 2])

alpha <- apply(la$alpha, 2:3, mean)
alpha.lb <- apply(la$alpha, 2:3, quantile, probs = 0.025)
alpha.ub <- apply(la$alpha, 2:3, quantile, probs = 0.975)

futurePredictedY_mean <- apply(futurePredictedY, 1, mean)
futurePredictedY.lb <- apply(futurePredictedY, 1, quantile, probs = 0.025)
futurePredictedY.ub <- apply(futurePredictedY, 1, quantile, probs = 0.975)

betaObs <- apply(la$betaObs, 2:3, mean)
betaObs.lb <- apply(la$betaObs, 2:3, quantile, probs = 0.025)
betaObs.ub <- apply(la$betaObs, 2:3, quantile, probs = 0.975)

# plot alpha over time and compare with true values
pdf("smoothed_time_series.pdf")
for (i in 1:1) { # dim(la$alpha)[3]
    plot(1:dim(la$alpha)[2], alpha[,i], ylim = c(min(alpha.lb[,i]), max(alpha.ub[,i])), type = 'l', ylab = "y", xlab = "time"); 
    points(1:dim(la$alpha)[2], y = alpha.lb[,i], lty = 'dotted', type = 'l');
    points(1:dim(la$alpha)[2], y = alpha.ub[,i], lty = 'dotted', type = 'l');
}
points(y_ts[1, ], type = 'p', col = q4[2])
points(y_ts_smooth, type = 'l', col = q4[2])
legend("bottomleft", lty = 1, legend = c("time series and its moving average (as above)", "smoothed times series (with 95%-credibility interval)"), col = q4[c(2, 1)])
abline(v = 70)
text(x = c(35, 85), y = 400, labels = c("Training", "Test"))
dev.off()

pdf("compare_predictions_with_simulated_data.pdf")
plot(x = seq(71, 100, length = 150), y = futurePredictedY_mean, ylim = c(-50, 1500), col = q4[4], xlab = "time", ylab = "y", main = "Predictions during test period")
segments(x0 = seq(71, 100, length = 150), x1 = seq(71, 100, length = 150), col = q4[4], y0 = futurePredictedY.lb, y1 = futurePredictedY.ub)
points(seq(71, 100, length = 150), dt_test$y)
dev.off()

pdf("coefficients_over_time.pdf")
par(mfrow = c(2,1))
plot(1:dim(la$alpha)[2], alpha[,2], ylim = c(min(alpha.lb[,2]), max(alpha.ub[,2])), type = 'l', ylab = expression(b[1]), xlab = "time"); 
legend("topleft", lty = 1, legend = c("true coefficient", "estimate (with 95%-credibility interval)"), col = q4[c(2, 1)])
points(1:dim(la$alpha)[2], y = alpha.lb[,2], lty = 'dotted', type = 'l');
points(1:dim(la$alpha)[2], y = alpha.ub[,2], lty = 'dotted', type = 'l');
abline(h = 30, lty = "dotdash", col = q4[2])
abline(v = 70)
text(x = c(35, 85), y = 27.5, labels = c("Training", "Test"))

plot(1:dim(la$alpha)[2], alpha[,3], ylim = c(min(alpha.lb[,3]), max(alpha.ub[,3])), type = 'l', ylab = expression(b[2]), xlab = "time"); 
points(1:dim(la$alpha)[2], y = alpha.lb[,3], lty = 'dotted', type = 'l');
points(1:dim(la$alpha)[2], y = alpha.ub[,3], lty = 'dotted', type = 'l');
abline(h = 0, lty = "dotdash", col = q4[2])
abline(v = 70)
text(x = c(35, 85), y = -0.15, labels = c("Training", "Test"))
dev.off()

####################################
# test binary outcomes. y_binary1 <- as.integer(dt$y > dt$ts * dt$x0 + 30 * dt$x1)
####################################


numCovariates <- 3
dynamicModelData = list(
    includeTrend = 0,
    N = nrow(dt_train),
    p = numCovariates,
    timePeriods = length(unique(dt_train$t)),
    futureTimePeriods = 30, # for how many time periods in the future do we need predictions?
    s = data.table::data.table(dt_train)[, .N, by = .(t)]$N,
    m0 = rep(0, numCovariates),
    C0 = 1e+07 * diag(numCovariates),
    m0_nu = rep(0, numCovariates),
    C0_nu = 1e+07 * diag(numCovariates),
    x = data.frame(dt_train$x0, dt_train$x1, dt_train$x2),
    y = dt_train$y_binary1
)


dyn_fit <- stan(file = "dynamic_model_binary_outcome_with_trend_reparameterized.stan", data = dynamicModelData, iter=2000, chains=8, cores = 8,
#               no progress after 10 iterations with the following: control = list(max_treedepth = 15, adapt_delta = 0.9)
               control = list(adapt_delta = 0.9, stepsize = 0.8), # one might also try: algorithm = 'HMC'
                open_progress = FALSE, verbose = TRUE, refresh = 100)

la <- extract(dyn_fit, permuted = TRUE) # return a list of arrays

ndraws <- 8000
# get predicted future values for betaObs and for predicted y
futureBetaObs <- array(NA_real_, c(ndraws, 100, 3))
futurePredictedY <- array(NA_real_, c(nrow(dt_test_X), ndraws))
for (t in 71:100) {
    for (p in 1:3) {
        futureBetaObs[, t, p] <- rnorm(ndraws, la$alpha[, t, p], la$sigma_betaObs[, p])
    }

    for (draw in 1:ndraws) {
        if (length(which(dt_test$t == t)) > 0) { # run the following only if we have covariates at time t
            futurePredictedY[which(dt_test$t == t), draw] <- as.matrix(dt_test_X[which(dt_test$t == t), c("x0", "x1", "x2")]) %*% la$betaObs[draw, t, ]
        }
    }
}

## few signs that chains did not mix..
plot(futurePredictedY[49,])
plot(la$betaObs[, 30, 2])


alpha <- apply(la$alpha, 2:3, mean)
alpha.lb <- apply(la$alpha, 2:3, quantile, probs = 0.025)
alpha.ub <- apply(la$alpha, 2:3, quantile, probs = 0.975)

futurePredictedY_mean <- apply(futurePredictedY, 1, mean)
futurePredictedY.lb <- apply(futurePredictedY, 1, quantile, probs = 0.025)
futurePredictedY.ub <- apply(futurePredictedY, 1, quantile, probs = 0.975)

betaObs <- apply(la$betaObs, 2:3, mean)
betaObs.lb <- apply(la$betaObs, 2:3, quantile, probs = 0.025)
betaObs.ub <- apply(la$betaObs, 2:3, quantile, probs = 0.975)

# plot alpha over time and compare with true values
pdf("smoothed_time_series_binary1.pdf")
for (i in 1:1) { # dim(la$alpha)[3]
    plot(1:dim(la$alpha)[2], alpha[,i], ylim = c(min(alpha.lb[,i]), max(alpha.ub[,i])), type = 'l', ylab = "y", xlab = "time"); 
    points(1:dim(la$alpha)[2], y = alpha.lb[,i], lty = 'dotted', type = 'l');
    points(1:dim(la$alpha)[2], y = alpha.ub[,i], lty = 'dotted', type = 'l');
}
points(y_ts[1, ], type = 'p', col = 2)
points(y_ts_smooth, type = 'l', col = 2)
legend("bottomleft", lty = 1, legend = c("time series and its moving average (as above)", "smoothed times series (with 95%-credibility interval)"), col = c(2, 1))
abline(v = 70)
text(x = c(35, 85), y = 400, labels = c("Training", "Test"))
dev.off()

pdf("compare_predictions_with_simulated_data_binary1.pdf")
plot(x = seq(71, 100, length = 150), y = futurePredictedY_mean, ylim = c(0, 1), col = 4, xlab = "time", ylab = "y", main = "Predictions during test period")
segments(x0 = seq(71, 100, length = 150), x1 = seq(71, 100, length = 150), col = 4, y0 = futurePredictedY.lb, y1 = futurePredictedY.ub)
points(seq(71, 100, length = 150), dt_test$y_binary1)
dev.off()

pdf("coefficients_over_time_binary1.pdf")
par(mfrow = c(2,1))
plot(1:dim(la$alpha)[2], alpha[,2], ylim = c(min(alpha.lb[,2]), max(alpha.ub[,2])), type = 'l', ylab = expression(b[1]), xlab = "time"); 
legend("topleft", lty = 1, legend = c("true coefficient", "estimate (with 95%-credibility interval)"), col = c(2, 1))
points(1:dim(la$alpha)[2], y = alpha.lb[,2], lty = 'dotted', type = 'l');
points(1:dim(la$alpha)[2], y = alpha.ub[,2], lty = 'dotted', type = 'l');
abline(h = 30, lty = "dotdash", col = 2)
abline(v = 70)
text(x = c(35, 85), y = 27.5, labels = c("Training", "Test"))

plot(1:dim(la$alpha)[2], alpha[,3], ylim = c(min(alpha.lb[,3]), max(alpha.ub[,3])), type = 'l', ylab = expression(b[2]), xlab = "time"); 
points(1:dim(la$alpha)[2], y = alpha.lb[,3], lty = 'dotted', type = 'l');
points(1:dim(la$alpha)[2], y = alpha.ub[,3], lty = 'dotted', type = 'l');
abline(h = 0, lty = "dotdash", col = 2)
abline(v = 70)
text(x = c(35, 85), y = -0.15, labels = c("Training", "Test"))
dev.off()


####################################
# test binary outcomes. y_binary3 <- as.integer(dt$y > 900)
####################################

numCovariates <- 3
dynamicModelData = list(
    includeTrend = 0,
    N = nrow(dt_train),
    p = numCovariates,
    timePeriods = length(unique(dt_train$t)),
    futureTimePeriods = 30, # for how many time periods in the future do we need predictions?
    s = data.table::data.table(dt_train)[, .N, by = .(t)]$N,
    m0 = rep(0, numCovariates),
    C0 = 1e+07 * diag(numCovariates),
    m0_nu = rep(0, numCovariates),
    C0_nu = 1e+07 * diag(numCovariates),
    x = data.frame(dt_train$x0, dt_train$x1, dt_train$x2),
    y = dt_train$y_binary3
)


dyn_fit <- stan(file = "dynamic_model_binary_outcome_with_trend_reparameterized.stan", data = dynamicModelData, iter=2000, chains=8, cores = 8,
#               no progress after 10 iterations with the following: control = list(max_treedepth = 15, adapt_delta = 0.9)
               control = list(adapt_delta = 0.9, stepsize = 0.8), # one might also try: algorithm = 'HMC'
                open_progress = FALSE, verbose = TRUE, refresh = 100)

la <- extract(dyn_fit, permuted = TRUE) # return a list of arrays

ndraws <- 8000
# get predicted future values for betaObs and for predicted y
futureBetaObs <- array(NA_real_, c(ndraws, 100, 3))
futurePredictedY <- array(NA_real_, c(nrow(dt_test_X), ndraws))
for (t in 71:100) {
    for (p in 1:3) {
        futureBetaObs[, t, p] <- rnorm(ndraws, la$alpha[, t, p], la$sigma_betaObs[, p])
    }

    for (draw in 1:ndraws) {
        if (length(which(dt_test$t == t)) > 0) { # run the following only if we have covariates at time t
            futurePredictedY[which(dt_test$t == t), draw] <- as.matrix(dt_test_X[which(dt_test$t == t), c("x0", "x1", "x2")]) %*% la$betaObs[draw, t, ]
        }
    }
}

## chains did not mix.. results cannot be trusted
plot(futurePredictedY[49,])
plot(la$betaObs[, 30, 2])
