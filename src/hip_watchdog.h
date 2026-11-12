#pragma once

// Host-only watchdog. Never calls the GPU driver: even a driver call used to
// query completion can itself hang. Long idle periods are not GPU stalls.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace hipwatch {
using Clock = std::chrono::steady_clock;
struct Call {
  const char* file = nullptr;
  const char* kind = nullptr;
  int line = 0;
  Clock::time_point since{};
};
struct Slot {
  std::mutex mutex;
  Call call;
};

class Watchdog {
 public:
  using Callback = void (*)(const Call&);
  Watchdog() = default;
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  // Call once, before serving. A zero timeout disables tracking.
  void start(std::chrono::milliseconds timeout, Callback callback) {
    if (timeout.count() <= 0) return;
    thread_ = std::thread([this, timeout, callback] {
      std::unique_lock<std::mutex> lock(mutex_);
      while (!wake_.wait_for(lock, std::chrono::milliseconds(100),
                             [this] { return stop_; })) {
        for (auto it = slots_.begin(); it != slots_.end();) {
          auto slot = it->lock();
          if (!slot) { it = slots_.erase(it); continue; }
          ++it;
          Call expired;
          {
            std::lock_guard<std::mutex> call_lock(slot->mutex);
            if (slot->call.file && Clock::now() - slot->call.since >= timeout)
              expired = slot->call;
          }
          if (expired.file) {
            enabled_.store(false, std::memory_order_release);
            lock.unlock();
            callback(expired);
            return;  // one report, even if a test callback returns
          }
        }
      }
    });
    enabled_.store(true, std::memory_order_release);
  }
  bool enabled() const { return enabled_.load(std::memory_order_acquire); }
  std::shared_ptr<Slot> add_slot() {
    auto slot = std::make_shared<Slot>();
    std::lock_guard<std::mutex> lock(mutex_);
    slots_.push_back(slot);
    return slot;
  }

 private:
  std::atomic<bool> enabled_{false};
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stop_ = false;
  std::vector<std::weak_ptr<Slot>> slots_;
  std::thread thread_;
};

inline Watchdog& instance() {
  static Watchdog watchdog;
  return watchdog;
}

class Scope {
 public:
  Scope(const char* file, int line, const char* kind) {
    if (!instance().enabled()) return;
    // One registration per host thread; expired weak entries are reclaimed.
    static thread_local auto local = instance().add_slot();
    slot_ = local.get();
    std::lock_guard<std::mutex> lock(slot_->mutex);
    previous_ = slot_->call;
    // Preserve the outer deadline for nested wrappers.
    if (!previous_.file) slot_->call = {file, kind, line, Clock::now()};
  }
  ~Scope() {
    if (!slot_) return;
    std::lock_guard<std::mutex> lock(slot_->mutex);
    slot_->call = previous_;
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
 private:
  Slot* slot_ = nullptr;
  Call previous_;
};
}  // namespace hipwatch
