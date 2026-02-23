// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/*
   __VIVADO_HLS_COPYRIGHT-INFO__
 
*/

#ifndef X_HLS_TASK_H
#define X_HLS_TASK_H
#include <functional>

#define HLS_TASK hls_thread_local hls::task
#define HLS_TASK_STREAM hls_thread_local hls::stream
#define HLS_TASK_STREAM_OF_BLOCKS hls_thread_local hls::stream_of_blocks
#define HLS_TASK_MERGE hls_thread_local hls::merge
#define HLS_TASK_SPLIT hls_thread_local hls::split

#ifdef __SYNTHESIS__
#include "hls_stream.h"
#include "hls_streamofblocks.h"

#else
// hls_task requires multi-thread version of hls::stream
#if defined(X_HLS_STREAM_SIM_H) && !defined(HLS_STREAM_THREAD_SAFE)
#error "Don't include thread unsafe hls_stream.h before hls_task.h"
#endif
#if defined(X_HLS_STREAMOFBLOCKS_H) && !defined(HLS_STREAM_THREAD_SAFE)
#error "Don't include thread unsafe hls_streamofbloks.h before hls_task.h"
#endif
#if defined(X_HLS_NP_CHANNEL_H) && !defined(HLS_STREAM_THREAD_SAFE)
#error "Don't include thread unsafe hls_np_channel.h before hls_task.h"
#endif
#if !defined(__HLS_COSIM__) && defined(__VITIS_HLS__)
#error "hls::tasks are not supported in bcsim"
#endif

#define HLS_STREAM_THREAD_SAFE
#include "hls_stream.h"
#include "hls_streamofblocks.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#endif // __SYNTHESIS__

namespace hls {

template<typename T>
struct is_hls_streaming_type : std::false_type {};

template<typename T, int DEPTH>
struct is_hls_streaming_type<hls::stream<T, DEPTH>&>: std::true_type {};

template<typename T, int DEPTH, int SIZE>
struct is_hls_streaming_type<hls::stream<T, DEPTH>(&)[SIZE]>: std::true_type {};

template<typename T, int DEPTH>
struct is_hls_streaming_type<hls::stream_of_blocks<T, DEPTH>&>: std::true_type {};

template<typename T, int DEPTH, int SIZE>
struct is_hls_streaming_type<hls::stream_of_blocks<T, DEPTH>(&)[SIZE]>: std::true_type {};

template < typename... List >
struct good_types_for_hls_task : std::true_type {};

template < typename Head, typename... Rest >
struct good_types_for_hls_task<Head, Rest...>
: std::conditional<is_hls_streaming_type<Head>::value && good_types_for_hls_task<Rest...>::value,
    std::true_type,
    std::false_type
>::type {};

template <>
struct good_types_for_hls_task<> : std::true_type {};

#ifdef __SYNTHESIS__
class task {
  int tag;
public:
  task() {
  }
  template <class T, class... Args>
  void operator()(T &&fn, Args&&... args) {
  #pragma HLS inline
     __attribute__((xlx_infinite_task(&tag))) fn(args...);
  }
  template <class T, class... Args>
  task(T &&fn, Args&&... args) {
  #pragma HLS inline
     __attribute__((xlx_infinite_task(&tag))) fn(args...);
  }
};

#define hls_thread_local

#else 
class task {
public:
  task() {
    stream_globals::incr_task_counter();
  }
  ~task() {
  }
  template <class T, class... Args>
  void operator()(T fn, Args&&... args) {
#if !defined(HLS_TASK_ALLOW_NON_STREAM_ARGS) && defined(__HLS_CSIM__) 
    static_assert(good_types_for_hls_task<Args...>::value, "All arguments of hls::task must be either of type hls::stream or hls::stream_of_blocks or an array thereof.");
#endif
    auto t_wrapper = [&] (T fn1, Args... args1) { // A wrapper to add while(1) around the real task function.
      while(1) {
        fn1(args1...);
      }
    };
    std::thread tmp(t_wrapper, fn, std::ref(args)...);
    t = std::move(tmp);
    t.detach();
  }
  template <class T, class... Args>
  task(T fn, Args&&... args) {
#if !defined(HLS_TASK_ALLOW_NON_STREAM_ARGS) && defined(__HLS_CSIM__) 
    static_assert(good_types_for_hls_task<Args...>::value, "All arguments of hls::task must be either of type hls::stream or hls::stream_of_blocks or an array thereof.");
#endif
    stream_globals::incr_task_counter();
    auto t_wrapper = [&] (T fn1, Args... args1) { // A wrapper to add while(1) around the real task function.
      while(1) {
        fn1(args1...);
      }
    };
    std::thread tmp(t_wrapper, fn, std::ref(args)...);
    t = std::move(tmp);
    t.detach();
  }
private:
  std::thread t;
};

#define hls_thread_local thread_local
#endif // __SYNTHESIS__
} 
#endif
