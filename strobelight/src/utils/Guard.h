// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
#include <functional>

class Guard {
 public:
  explicit Guard(std::function<void()> cleanupFunction)
      : cleanupFunction_(std::move(cleanupFunction)) {}

  ~Guard() {
    if (cleanupFunction_) {
      cleanupFunction_();
    }
  }

  // Disable copy and move operations to prevent unintended behavior
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;
  Guard(Guard&&) = delete;
  Guard& operator=(Guard&&) = delete;

 private:
  std::function<void()> cleanupFunction_;
};
