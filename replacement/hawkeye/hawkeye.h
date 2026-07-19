#ifndef REPLACEMENT_HAWKEYE_H
#define REPLACEMENT_HAWKEYE_H

#include <cstdint>
#include <vector>

#include "cache.h"
#include "modules.h"

struct hawkeye_line {
  bool valid = false;
  uint64_t tag = 0;
  uint64_t pc = 0;
  int rrpv = 7; // 3-bit RRIP value: higher = easier to evict
};

struct sampler_entry {
  bool valid = false;
  uint64_t tag = 0;
  uint64_t pc = 0;
  uint64_t last_time = 0;
};

struct hawkeye_set {
  std::vector<hawkeye_line> lines;

  explicit hawkeye_set(long ways);

  long victim();
};

// Hardware budget for the 2 MB / 16-way / 2048-set LLC simulated here,
// fitting the 32 KB budget used for all non-LRU policies:
//   RRPV:       32768 lines x 3 bits                            = 12.00 KB
//   Predictor:  8192 entries x 3 bits                           =  3.00 KB
//   Sampler:    64 sampled sets x 38 entries x (1 valid
//               + 16 tag + 13 PC signature + 8 timestamp) bits  = 11.28 KB
//   Occupancy vectors: 64 sets x 32 buckets x 5 bits            =  1.25 KB
//   Per-line PC signature, sampled sets only (used to detrain
//   on eviction): 64 sets x 16 ways x 13 bits                   =  1.63 KB
//   Set timers: 64 x 12 bits                                    =  0.09 KB
//   Total                                                       = 29.25 KB
// (The simulator stores full addresses/PCs/timers for convenience; the widths
// above are the hardware-equivalent field sizes the policy relies on.)
struct hawkeye : public champsim::modules::replacement {

  static constexpr int MAX_RRIP    = 7;
  static constexpr int PRED_SIZE   = 8192;
  static constexpr int PRED_MAX    = 7;
  static constexpr int PRED_INIT   = 4;   // midpoint of [0,7]
  static constexpr int PRED_THRESH = 4;   // >= friendly, < averse
  static constexpr int OCC_SIZE    = 32;
  static constexpr int TIME_QUANTUM = 4;
  static constexpr int SAMPLER_SIZE = 38;

  std::vector<hawkeye_set> sets;

  // PC predictor: 3-bit counters, 0-3 = averse, 4-7 = friendly
  std::vector<int> predictor;

  // Sampled cache for OPTgen
  std::vector<std::vector<sampler_entry>> sampler;

  // Occupancy vector for sampled sets
  std::vector<std::vector<int>> occupancy;

  std::vector<uint64_t> set_timer;

  // Track last bucket seen per set so we only zero on bucket entry
  std::vector<uint64_t> last_occ_bucket;

  explicit hawkeye(CACHE* cache);
  hawkeye(CACHE* cache, long sets_, long ways_);

  bool is_sampled_set(long set) const;
  uint32_t hash(uint64_t pc);

  bool predict(uint64_t pc);
  void train(uint64_t pc, bool opt_hit);

  void optgen_access(long set, uint64_t addr, uint64_t pc);

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set,
                   const champsim::cache_block* current_set,
                   champsim::address ip, champsim::address full_addr,
                   access_type type);

  // Fill path (misses)
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way,
                               champsim::address full_addr,
                               champsim::address ip,
                               champsim::address victim_addr,
                               access_type type);

  // Hit path only
  void update_replacement_state(uint32_t triggering_cpu, long set, long way,
                                champsim::address full_addr,
                                champsim::address ip,
                                champsim::address victim_addr,
                                access_type type, uint8_t hit);
};

#endif