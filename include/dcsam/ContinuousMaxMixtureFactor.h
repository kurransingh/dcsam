/**
 * @file ContinuousMaxMixtureFactor.h
 * @brief Continuous Only Max-Mixture factor that plays nicely with DCSAM
 * @author Kurran Singh, singhk@mit.edu
 *
 * Copyright 2023 The Ambitious Folks of the MRG
 */

#pragma once

#include <math.h>

#include <algorithm>
#include <limits>
#include <vector>

#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dcsam {

/**
 * @brief Implementation of a discrete-continuous max-mixture factor
 *
 * r(x) = min_i -log(w_i) + r_i(x)
 *
 * The error returned from this factor is the minimum error + weight
 * over all of the component factors
 * See Olson and Agarwal RSS 2012 for details
 */
template <class ContinuousFactorType>
class ContinuousMaxMixtureFactor : public gtsam::NonlinearFactor {
 private:
  std::vector<ContinuousFactorType> factors_;
  std::vector<double> log_weights_;
  bool normalized_;

 public:
  using Base = gtsam::NonlinearFactor;

  ContinuousMaxMixtureFactor() = default;

  explicit ContinuousMaxMixtureFactor(const std::vector<ContinuousFactorType> factors,
                              const std::vector<double> weights,
                              const bool normalized)
      : Base(), normalized_(normalized) {
    factors_ = factors;
    for (size_t i = 0; i < weights.size(); i++) {
      log_weights_.push_back(log(weights[i]));
    }
  }

  explicit ContinuousMaxMixtureFactor(const std::vector<ContinuousFactorType> factors,
                              const bool normalized)
      : Base(), normalized_(normalized) {
    factors_ = factors;
    for (size_t i = 0; i < factors_.size(); i++) {
      log_weights_.push_back(0);
    }
  }

  ContinuousMaxMixtureFactor& operator=(const ContinuousMaxMixtureFactor& rhs) {
    this->factors_ = rhs.factors_;
    this->log_weights_ = rhs.log_weights_;
    this->normalized_ = rhs.normalized_;
  }

  virtual ~ContinuousMaxMixtureFactor() = default;

  double error(const gtsam::Values& continuousVals) const override {
    size_t min_error_idx = getActiveFactorIdx(continuousVals);
    assert(0 <= min_error_idx);
    assert(min_error_idx < factors_.size());
    double min_error =
        factors_[min_error_idx].error(continuousVals);
    // if (normalized_) {
    return min_error - log_weights_[min_error_idx]; //}
    // return min_error +
    //        factors_[min_error_idx].logNormalizingConstant(continuousVals) -
    //        log_weights_[min_error_idx];
  }

  size_t getActiveFactorIdx(const gtsam::Values& continuousVals) const {
    double min_error = std::numeric_limits<double>::infinity();
    size_t min_error_idx = 0;
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
    if (factors_.size() > 0) {
      return factors_[0].dim();
    } else {
      return 0;
    }
  }

  bool equals(const gtsam::NonlinearFactor& other, double tol = 1e-9) const override {
    if (!dynamic_cast<const ContinuousMaxMixtureFactor*>(&other)) return false;
    const ContinuousMaxMixtureFactor& f(static_cast<const ContinuousMaxMixtureFactor&>(other));
    if (factors_.size() != f.factors_.size()) return false;
    // for (size_t i = 0; i < factors_.size(); i++) {
    //   if (!factors_[i].equals(f.factors_[i])) return false;
    // }
    return ((log_weights_ == f.log_weights_) && (normalized_ == f.normalized_));
  }

  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& continuousVals) const override {
    size_t min_error_idx = getActiveFactorIdx(continuousVals);
    return factors_[min_error_idx].linearize(continuousVals);
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
    for (int i = 0; i < weights.size(); i++) {
      log_weights_[i] = log(weights[i]);
    }
  }
};
}  // namespace dcsam
