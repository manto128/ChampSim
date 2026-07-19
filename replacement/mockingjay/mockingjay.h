#ifndef REPLACEMENT_MOCKINGJAY_H
#define REPLACEMENT_MOCKINGJAY_H

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "modules.h"

// Mockingjay replacement policy.
//
// I. Shah, A. Jain, C. Lin, "Effective Mimicry of Belady's MIN Policy",
// HPCA 2022. Follow-up to Hawkeye: instead of a binary friendly/averse
// classification, a PC-indexed Reuse Distance Predictor (RDP) predicts each
// line's reuse distance, and the line whose predicted reuse lies furthest in
// the future (largest |ETR|) is evicted. Fills predicted to be reused later
// than every resident line are bypassed.
//
// Hardware budget for the 2 MB / 16-way / 2048-set LLC with 64 B lines
// simulated here (paper Section III-E), fitting the 32 KB budget:
//   ETR counters:      32768 lines x 5 bits                     = 20.00 KB
//   ETR aging clocks:  2048 sets x 3 bits                       =  0.75 KB
//   RDP:               2^11 entries x 7 bits                    =  1.75 KB
//   Sampled cache:     32 sampled sets x 16 sub-sets x 5 ways
//                      = 2560 entries x (1 valid + 10 tag
//                        + 11 signature + 8 timestamp) bits     =  9.38 KB
//   Sampled-set timestamp counters: 32 x 8 bits                 =  0.03 KB
//   Total                                                       = 31.91 KB
// (The simulator stores wider values for convenience; the widths above are
// the hardware-equivalent field sizes the policy actually relies on.)

struct mockingjay_sampled_entry {
  bool valid = false;
  uint64_t tag = 0;       // SAMPLED_CACHE_TAG_BITS-wide block address hash
  uint64_t signature = 0; // PC_SIGNATURE_BITS-wide PC signature
  int timestamp = 0;      // TIMESTAMP_BITS-wide coarse timestamp
};

struct mockingjay : public champsim::modules::replacement {

  // Fixed design parameters (paper Section III-B)
  static constexpr int HISTORY = 8;                 // history length: 8x the set size
  static constexpr int GRANULARITY = 8;             // ETR aging period f: one aging per 8 set accesses
  static constexpr int SAMPLED_CACHE_WAYS = 5;
  static constexpr int LOG2_SAMPLED_CACHE_SETS = 4; // 16 sampled-cache sub-sets per sampled LLC set
  static constexpr int TIMESTAMP_BITS = 8;
  static constexpr double TEMP_DIFFERENCE = 1.0 / 16.0; // temporal-difference learning rate

  long NUM_SET, NUM_WAY;

  // Parameters derived from the cache geometry (2 MB 16-way LLC values in comments)
  int LOG2_BLOCK_SIZE;        // 6
  int LOG2_LLC_SET;           // 11
  int LOG2_LLC_SIZE;          // 21
  int LOG2_SAMPLED_SETS;      // 5  -> 32 sampled sets
  int INF_RD;                 // 127: reuse distance representing "never reused" (scan)
  int MAX_RD;                 // 105: predictions above this are treated as scans
  int INF_ETR;                // 15: ETR pinning scanning lines at highest eviction priority
  int SAMPLED_CACHE_TAG_BITS; // 10
  int PC_SIGNATURE_BITS;      // 11
  double FLEXMIN_PENALTY;     // 2.0 on one core: reuse-distance inflation for prefetch intervals

  std::vector<int> etr;           // per-line Estimated Time Remaining counter [-INF_ETR, INF_ETR]
  std::vector<int> etr_clock;     // per-set aging clock
  std::vector<int> set_timestamp; // per-set access counter, used only for sampled sets

  // Reuse Distance Predictor: PC signature -> predicted reuse distance [0, INF_RD]
  std::unordered_map<uint64_t, int> rdp;

  // Sampled cache holding the access history of the sampled sets,
  // indexed by (sub-set bits . LLC set bits) of the block address
  std::unordered_map<uint32_t, std::vector<mockingjay_sampled_entry>> sampled_cache;

  explicit mockingjay(CACHE* cache);

  bool is_sampled_set(long set) const;
  uint64_t get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core) const;
  uint32_t get_sampled_cache_index(uint64_t full_addr) const;
  uint64_t get_sampled_cache_tag(uint64_t full_addr) const;
  int search_sampled_cache(uint64_t tag, uint32_t index) const;
  void detrain(uint32_t index, int way);

  int temporal_difference(int init, int sample) const;
  int increment_timestamp(int input) const;
  int time_elapsed(int global, int local) const;

  int& etr_of(long set, long way);

  // Common path for demand/prefetch hits and fills (way == NUM_WAY on a bypass)
  void update(uint32_t cpu, long set, long way, champsim::address full_addr,
              champsim::address ip, access_type type, bool hit);

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set,
                   const champsim::cache_block* current_set,
                   champsim::address ip, champsim::address full_addr,
                   access_type type);

  // Fill path (misses); way == NUM_WAY indicates a bypassed fill
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
