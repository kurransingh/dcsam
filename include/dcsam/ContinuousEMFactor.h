/**
 * @file ContinuousEMFactor.h.h
 * @brief Continuous EM factor
 * @author Kurran Singh, singhk@mit.edu
 * Copyright 2023 The Ambitious Folks of the MRG
 */

#pragma once

#include <math.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include "dcsam/DCSAM_utils.h"

namespace dcsam {

/**
 * @brief Implementation of a continuous EM factor
 *
 * The error function is defined as:
 * r(x) = sum_i w'_i * r_i(x),
 *
 * where
 * w'_i = w_i * p(z | x, h_i); sum_i w'_i = 1.
 * and h_i here represents the "i-th" hypothesis
 *
 * The error returned from this factor is a weighted combination of the
 * component factor errors. "x" can be comprised jointly of discrete and
 * continuous values. Prior hypothesis weights can be included if desired via
 * the `weights` parameter.
 */
template <class ContinuousFactorType>
class ContinuousEMFactor : public gtsam::NonlinearFactor {
 private:
  std::vector<ContinuousFactorType> factors_;
  std::vector<double> log_weights_;
  bool normalized_;

 public:
  using Base = gtsam::NonlinearFactor;

  ContinuousEMFactor() = default;

  explicit ContinuousEMFactor(const gtsam::KeyVector& keys,
                      const std::vector<ContinuousFactorType> factors,
                      const std::vector<double> weights, const bool normalized)
      : Base(keys), normalized_(normalized) {
    factors_ = factors;
    for (size_t i = 0; i < weights.size(); i++) {
      log_weights_.push_back(log(weights[i]));
    }
  }

  explicit ContinuousEMFactor(const std::vector<ContinuousFactorType> factors,
                      const bool normalized)
      : Base(), normalized_(normalized) {
    factors_ = factors;
    for (size_t i = 0; i < factors_.size(); i++) {
      log_weights_.push_back(0);
    }
  }

  ContinuousEMFactor& operator=(const ContinuousEMFactor& rhs) {
    this->factors_ = rhs.factors_;
    this->log_weights_ = rhs.log_weights_;
    this->normalized_ = rhs.normalized_;
  }

  virtual ~ContinuousEMFactor() = default;

  double error(const gtsam::Values& continuousVals) const override {
    // Retrieve the log prob for each component.
    std::vector<double> logprobs =
        computeComponentLogProbs(continuousVals);

    // Weights for each component are obtained by normalizing the errors.
    std::vector<double> componentWeights = expNormalize(logprobs);

    // Compute the total error as the weighted sum of component errors.
    double total_error = 0.0;
    for (size_t i = 0; i < logprobs.size(); i++) {
      total_error += componentWeights[i] * (-logprobs[i]);
    }
    return total_error;
  }

  std::vector<double> computeComponentLogProbs(
      const gtsam::Values& continuousVals) const {
    // Container for errors, where:
    //   error_i = error of component factor i - log_weights_i
    std::vector<double> logprobs;
    for (size_t i = 0; i < factors_.size(); i++) {
      double error =
          factors_[i].error(continuousVals) - log_weights_[i];
      // if (!normalized_)
      //   error += factors_[i].logNormalizingConstant(continuousVals);
      logprobs.push_back(-error);
    }
    return logprobs;
  }

  size_t getActiveFactorIdx(const gtsam::Values& continuousVals) const {
    double min_error = std::numeric_limits<double>::infinity();
    size_t min_error_idx;
    for (size_t i = 0; i < factors_.size(); i++) {
      double error =
          factors_[i].error(continuousVals) - log_weights_[i];
      // if (!normalized_)
      //   error += factors_[i].logNormalizingConstant(continuousVals);

      if (error < min_error) {
        min_error = error;
        min_error_idx = i;
      }
    }
    return min_error_idx;
  }

  size_t dim() const override {
    size_t total = 0;
    // Each component factor `i` requires `factors_[i].dim()` rows in the
    // overall Jacobian.
    for (size_t i = 0; i < factors_.size(); i++) {
      total += factors_[i].dim();
    }
    return total;
  }

  bool equals(const gtsam::NonlinearFactor& other, double tol = 1e-9) const override {
    if (!dynamic_cast<const ContinuousEMFactor*>(&other)) return false;
    const ContinuousEMFactor& f(static_cast<const ContinuousEMFactor&>(other));
    if (factors_.size() != f.factors_.size()) return false;
    // for (size_t i = 0; i < factors_.size(); i++) {
    //   if (!factors_[i].equals(f.factors_[i])) return false;
    // }
    return ((log_weights_ == f.log_weights_) && (normalized_ == f.normalized_));
  }

  /*
   * Jacobian magic
   */
  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& continuousVals) const override {
    std::vector<boost::shared_ptr<gtsam::GaussianFactor>> gfs;

    // Start by computing all errors, so we can get the component weights.
    std::vector<double> errors =
        computeComponentLogProbs(continuousVals);

    // Weights for each component are obtained by normalizing the errors.
    std::vector<double> componentWeights = expNormalize(errors);

    // We want to temporarily build a GaussianFactorGraph to construct the
    // Jacobian for this whole factor.
    gtsam::GaussianFactorGraph gfg;

    for (size_t i = 0; i < factors_.size(); i++) {
      // std::cout << "i = " << i << std::endl;
      // First get the GaussianFactor obtained by linearizing `factors_[i]`
      boost::shared_ptr<gtsam::GaussianFactor> gf =
          factors_[i].linearize(continuousVals);

      gtsam::JacobianFactor jf_component(*gf);

      // Recover the [A b] matrix with Jacobian A and right-hand side vector b,
      // with noise models "baked in," as a vertical block matrix.
      gtsam::VerticalBlockMatrix Ab = jf_component.matrixObject();

      // Copy Ab so we can reweight it appropriately.
      gtsam::VerticalBlockMatrix Ab_weighted = Ab;

      // Populate Ab_weighted with weighted Jacobian sqrt(w)*A and right-hand
      // side vector sqrt(w)*b.
      double sqrt_weight = sqrt(componentWeights[i]);

      for (size_t k = 0; k < Ab_weighted.nBlocks(); k++) {
        Ab_weighted(k) = sqrt_weight * Ab(k);
      }

      // Create a `JacobianFactor` from the system [A b] and add it to the
      // `GaussianFactorGraph`.
      gtsam::JacobianFactor jf(factors_[i].keys(), Ab_weighted);
      gfg.add(jf);
    }

    // Stack Jacobians to build combined factor.

    return boost::make_shared<gtsam::JacobianFactor>(gfg);
  }


  gtsam::FastVector<gtsam::Key> getAssociationKeys(
      const gtsam::Values& continuousVals) const {
    size_t min_error_idx = getActiveFactorIdx(continuousVals);
    return factors_[min_error_idx].keys();
  }

  void updateWeights(const std::vector<double>& weights) {
    if (weights.size() != log_weights_.size()) {
      std::cerr << "Attempted to update weights with incorrectly sized vector."
                << std::endl;
      return;
    }
    for (size_t i = 0; i < weights.size(); i++) {
      log_weights_[i] = log(weights[i]);
    }
  }
};
}  // namespace dcsam
