#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

template <typename T>
T timer() {
  using namespace std::chrono;
  auto now = high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto seconds = duration_cast<std::chrono::duration<T>>(duration);
  return seconds.count();
}

#endif  // TIMER_H_
