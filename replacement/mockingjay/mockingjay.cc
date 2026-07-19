#include "mockingjay.h"

#include <cmath>
#include <cstdlib>

#include "champsim.h"
#include "msl/bits.h"
#include "util/to_underlying.h"

// ---------------- HASHES AND INDEXING ----------------

namespace
{
// CRC-based hash used by the CRC2 framework and the Mockingjay reference code
uint64_t crc_hash(uint64_t block_address)
{
  constexpr uint64_t crc_polynomial = 3988292384ULL;
  uint64_t ret = block_address;
  for (unsigned i = 0; i < 3; i++)
    ret = ((ret & 1) == 1) ? ((ret >> 1) ^ crc_polynomial) : (ret >> 1);
  return ret;
}
} // namespace

// The RDP distinguishes hits from misses (and prefetches from demands) issued
// by the same PC, so those bits are folded into the hashed signature.
uint64_t mockingjay::get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core) const
{
  if (NUM_CPUS == 1) {
    pc = pc << 1;
    if (hit)
      pc = pc | 1;
    pc = pc << 1;
    if (prefetch)
      pc = pc | 1;
    pc = crc_hash(pc);
    pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
  } else {
    pc = pc << 1;
    if (prefetch)
      pc = pc | 1;
    pc = pc << 2;
    pc = pc | core;
    pc = crc_hash(pc);
    pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
  }
  return pc;
}

bool mockingjay::is_sampled_set(long set) const
{
  int mask_length = LOG2_LLC_SET - LOG2_SAMPLED_SETS;
  long mask = (1L << mask_length) - 1;
  return (set & mask) == ((set >> (LOG2_LLC_SET - mask_length)) & mask);
}

uint32_t mockingjay::get_sampled_cache_index(uint64_t full_addr) const
{
  full_addr = full_addr >> LOG2_BLOCK_SIZE;
  full_addr = (full_addr << (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET))) >> (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET));
  return static_cast<uint32_t>(full_addr);
}

uint64_t mockingjay::get_sampled_cache_tag(uint64_t full_addr) const
{
  full_addr >>= LOG2_LLC_SET + LOG2_BLOCK_SIZE + LOG2_SAMPLED_CACHE_SETS;
  full_addr = (full_addr << (64 - SAMPLED_CACHE_TAG_BITS)) >> (64 - SAMPLED_CACHE_TAG_BITS);
  return full_addr;
}

int mockingjay::search_sampled_cache(uint64_t tag, uint32_t index) const
{
  const std::vector<mockingjay_sampled_entry>& sampled_set = sampled_cache.at(index);
  for (int way = 0; way < SAMPLED_CACHE_WAYS; way++) {
    if (sampled_set[static_cast<std::size_t>(way)].valid && sampled_set[static_cast<std::size_t>(way)].tag == tag)
      return way;
  }
  return -1;
}

// ---------------- CONSTRUCTION ----------------

mockingjay::mockingjay(CACHE* cache) : replacement(cache), NUM_SET(cache->NUM_SET), NUM_WAY(cache->NUM_WAY)
{
  LOG2_BLOCK_SIZE = static_cast<int>(champsim::to_underlying(cache->OFFSET_BITS));
  LOG2_LLC_SET = static_cast<int>(champsim::msl::lg2(static_cast<uint64_t>(NUM_SET)));
  LOG2_LLC_SIZE = LOG2_LLC_SET + static_cast<int>(champsim::msl::lg2(static_cast<uint64_t>(NUM_WAY))) + LOG2_BLOCK_SIZE;
  LOG2_SAMPLED_SETS = LOG2_LLC_SIZE - 16; // one sampled set per 64 KB of capacity

  INF_RD = static_cast<int>(NUM_WAY) * HISTORY - 1;
  MAX_RD = INF_RD - 22;
  INF_ETR = (static_cast<int>(NUM_WAY) * HISTORY / GRANULARITY) - 1;

  SAMPLED_CACHE_TAG_BITS = 31 - LOG2_LLC_SIZE;
  PC_SIGNATURE_BITS = LOG2_LLC_SIZE - 10;
  FLEXMIN_PENALTY = 2.0 - std::log2(static_cast<double>(NUM_CPUS)) / 4.0;

  etr.assign(static_cast<std::size_t>(NUM_SET * NUM_WAY), 0);
  etr_clock.assign(static_cast<std::size_t>(NUM_SET), GRANULARITY);
  set_timestamp.assign(static_cast<std::size_t>(NUM_SET), 0);

  for (long set = 0; set < NUM_SET; set++) {
    if (is_sampled_set(set)) {
      uint32_t modifier = 1u << LOG2_LLC_SET;
      uint32_t limit = 1u << LOG2_SAMPLED_CACHE_SETS;
      for (uint32_t i = 0; i < limit; i++)
        sampled_cache[static_cast<uint32_t>(set) + modifier * i].resize(SAMPLED_CACHE_WAYS);
    }
  }
}

int& mockingjay::etr_of(long set, long way) { return etr[static_cast<std::size_t>(set * NUM_WAY + way)]; }

// ---------------- RDP TRAINING ----------------

// Temporal-difference update (paper Section III-B): the entry moves toward the
// observed reuse distance by at most 1, so a single outlier cannot destroy a
// well-established prediction.
int mockingjay::temporal_difference(int init, int sample) const
{
  if (sample > init) {
    int diff = sample - init;
    diff = static_cast<int>(diff * TEMP_DIFFERENCE);
    diff = std::min(1, diff);
    return std::min(init + diff, INF_RD);
  }
  if (sample < init) {
    int diff = init - sample;
    diff = static_cast<int>(diff * TEMP_DIFFERENCE);
    diff = std::min(1, diff);
    return std::max(init - diff, 0);
  }
  return init;
}

// A line leaving the sampled cache without being reused was a scan: nudge its
// PC's prediction toward INF_RD.
void mockingjay::detrain(uint32_t index, int way)
{
  if (way < 0)
    return;
  mockingjay_sampled_entry& entry = sampled_cache.at(index)[static_cast<std::size_t>(way)];
  if (!entry.valid)
    return;

  auto pred = rdp.find(entry.signature);
  if (pred != rdp.end())
    pred->second = std::min(pred->second + 1, INF_RD);
  else
    rdp[entry.signature] = INF_RD;
  entry.valid = false;
}

int mockingjay::increment_timestamp(int input) const { return (input + 1) % (1 << TIMESTAMP_BITS); }

int mockingjay::time_elapsed(int global, int local) const
{
  if (global >= local)
    return global - local;
  return global + (1 << TIMESTAMP_BITS) - local;
}

// ---------------- MAIN POLICY ----------------

long mockingjay::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                             champsim::address full_addr, access_type type)
{
  for (long way = 0; way < NUM_WAY; way++) {
    if (!current_set[way].valid)
      return way;
  }

  // Evict the line whose predicted reuse is furthest away: the largest |ETR|,
  // with ties broken in favor of lines already past their predicted reuse (ETR < 0)
  int max_etr = 0;
  long victim_way = 0;
  for (long way = 0; way < NUM_WAY; way++) {
    int candidate = etr_of(set, way);
    if (std::abs(candidate) > max_etr || (std::abs(candidate) == max_etr && candidate < 0)) {
      max_etr = std::abs(candidate);
      victim_way = way;
    }
  }

  // Bypass fills predicted to be reused later than every resident line
  // (writebacks may not bypass)
  if (type != access_type::WRITE) {
    uint64_t signature = get_pc_signature(ip.to<uint64_t>(), false, type == access_type::PREFETCH, triggering_cpu);
    auto pred = rdp.find(signature);
    if (pred != rdp.end() && (pred->second > MAX_RD || pred->second / GRANULARITY > max_etr))
      return NUM_WAY;
  }

  return victim_way;
}

void mockingjay::update(uint32_t cpu, long set, long way, champsim::address full_addr, champsim::address ip, access_type type, bool hit)
{
  uint64_t addr = full_addr.to<uint64_t>();
  uint64_t signature = get_pc_signature(ip.to<uint64_t>(), hit, type == access_type::PREFETCH, cpu);

  if (is_sampled_set(set)) {
    uint32_t sampled_cache_index = get_sampled_cache_index(addr);
    uint64_t sampled_cache_tag = get_sampled_cache_tag(addr);
    std::vector<mockingjay_sampled_entry>& sampled_set = sampled_cache.at(sampled_cache_index);

    // On a sampled-cache hit, the elapsed time is an observed reuse distance:
    // train the RDP entry of the PC that last touched the block
    int sampled_cache_way = search_sampled_cache(sampled_cache_tag, sampled_cache_index);
    if (sampled_cache_way > -1) {
      mockingjay_sampled_entry& entry = sampled_set[static_cast<std::size_t>(sampled_cache_way)];
      int sample = time_elapsed(set_timestamp[static_cast<std::size_t>(set)], entry.timestamp);
      if (sample <= INF_RD) {
        if (type == access_type::PREFETCH)
          sample = static_cast<int>(sample * FLEXMIN_PENALTY); // Flex-MIN: discourage caching of *-P intervals

        auto pred = rdp.find(entry.signature);
        if (pred != rdp.end())
          pred->second = temporal_difference(pred->second, sample);
        else
          rdp[entry.signature] = sample;

        entry.valid = false;
      }
    }

    // Free a way: entries older than the history window are scans, and if none
    // exists the LRU entry is detrained and replaced
    int lru_way = -1;
    int lru_rd = -1;
    for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
      if (!sampled_set[static_cast<std::size_t>(w)].valid) {
        lru_way = w;
        lru_rd = INF_RD + 1;
        continue;
      }
      int sample = time_elapsed(set_timestamp[static_cast<std::size_t>(set)], sampled_set[static_cast<std::size_t>(w)].timestamp);
      if (sample > INF_RD) {
        lru_way = w;
        lru_rd = INF_RD + 1;
        detrain(sampled_cache_index, w);
      } else if (sample > lru_rd) {
        lru_way = w;
        lru_rd = sample;
      }
    }
    detrain(sampled_cache_index, lru_way);

    for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
      mockingjay_sampled_entry& entry = sampled_set[static_cast<std::size_t>(w)];
      if (!entry.valid) {
        entry.valid = true;
        entry.tag = sampled_cache_tag;
        entry.signature = signature;
        entry.timestamp = set_timestamp[static_cast<std::size_t>(set)];
        break;
      }
    }

    set_timestamp[static_cast<std::size_t>(set)] = increment_timestamp(set_timestamp[static_cast<std::size_t>(set)]);
  }

  // Age the set once every GRANULARITY accesses; lines whose |ETR| has
  // saturated (scans and long-expired lines) are not aged
  if (etr_clock[static_cast<std::size_t>(set)] == GRANULARITY) {
    for (long w = 0; w < NUM_WAY; w++) {
      if (w != way && std::abs(etr_of(set, w)) < INF_ETR)
        etr_of(set, w)--;
    }
    etr_clock[static_cast<std::size_t>(set)] = 0;
  }
  etr_clock[static_cast<std::size_t>(set)]++;

  // Reset the accessed line's ETR from the RDP prediction (skipped on a bypass)
  if (way < NUM_WAY) {
    auto pred = rdp.find(signature);
    if (pred == rdp.end())
      etr_of(set, way) = (NUM_CPUS == 1) ? 0 : INF_ETR;
    else if (pred->second > MAX_RD)
      etr_of(set, way) = INF_ETR;
    else
      etr_of(set, way) = pred->second / GRANULARITY;
  }
}

void mockingjay::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                        champsim::address victim_addr, access_type type)
{
  if (type == access_type::WRITE) {
    // Writeback fills carry no PC: install with the highest eviction priority
    etr_of(set, way) = -INF_ETR;
    return;
  }
  update(triggering_cpu, set, way, full_addr, ip, type, false);
}

void mockingjay::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                          champsim::address victim_addr, access_type type, uint8_t hit)
{
  if (!hit)
    return; // misses are handled at fill time

  if (type == access_type::WRITE)
    return; // writeback hits leave the replacement state unchanged

  update(triggering_cpu, set, way, full_addr, ip, type, true);
}
