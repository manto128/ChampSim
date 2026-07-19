#include "hawkeye.h"

#include <algorithm>

// ---------------- SET ----------------

hawkeye_set::hawkeye_set(long ways) : lines(static_cast<std::size_t>(ways)) {}

long hawkeye_set::victim() {
  // Prefer invalid entries first
  for (long i = 0; i < static_cast<long>(lines.size()); i++) {
    if (!lines[static_cast<std::size_t>(i)].valid)
      return i;
  }

  // Prefer evicting a line with RRIP == MAX_RRIP 
  for (long i = 0; i < static_cast<long>(lines.size()); i++) {
    if (lines[static_cast<std::size_t>(i)].rrpv == hawkeye::MAX_RRIP)
      return i;
  }

  //evict the line with the highest RRIP

  long victim = 0;
  for (long i = 1; i < static_cast<long>(lines.size()); i++) {
    if (lines[static_cast<std::size_t>(i)].rrpv >
        lines[static_cast<std::size_t>(victim)].rrpv)
      victim = i;
  }

  return victim;
}

// ---------------- HAWKEYE ----------------

hawkeye::hawkeye(CACHE *cache)
    : hawkeye(cache, cache->NUM_SET, cache->NUM_WAY) {}

hawkeye::hawkeye(CACHE *cache, long sets_, long ways_) : replacement(cache) {
  sets.reserve(static_cast<std::size_t>(sets_));

  for (long i = 0; i < sets_; i++) {
    sets.emplace_back(ways_);
  }

  predictor.resize(PRED_SIZE, PRED_INIT);

  sampler.resize(static_cast<std::size_t>(sets_));
  occupancy.resize(static_cast<std::size_t>(sets_));
  set_timer.resize(static_cast<std::size_t>(sets_), 0);

  last_occ_bucket.resize(static_cast<std::size_t>(sets_), static_cast<uint64_t>(-1));

  for (long i = 0; i < sets_; i++) {
    if (is_sampled_set(i)) {
      sampler[static_cast<std::size_t>(i)].resize(SAMPLER_SIZE);
      occupancy[static_cast<std::size_t>(i)].resize(OCC_SIZE, 0);
    }
  }
}

bool hawkeye::is_sampled_set(long set) const {
  return (set % 32) == 0;
}

uint32_t hawkeye::hash(uint64_t pc) {
  return static_cast<uint32_t>((pc ^ (pc >> 2) ^ (pc >> 5)) & (PRED_SIZE - 1));
}

// ---------------- PREDICTOR ----------------

bool hawkeye::predict(uint64_t pc) {
  return predictor[hash(pc)] >= PRED_THRESH;
}

void hawkeye::train(uint64_t pc, bool opt_hit) {
  int &val = predictor[hash(pc)];

  if (opt_hit && val < PRED_MAX) {
    val++;
  } else if (!opt_hit && val > 0) {
    val--;
  }
}

// ---------------- OPTGEN WITH OCCUPANCY VECTOR ----------------

void hawkeye::optgen_access(long set, uint64_t addr, uint64_t pc) {
  if (!is_sampled_set(set))
    return;

  set_timer[static_cast<std::size_t>(set)]++;
  uint64_t timer = set_timer[static_cast<std::size_t>(set)];
  uint64_t current_bucket = (timer / TIME_QUANTUM) % OCC_SIZE;

  std::vector<sampler_entry> &samp = sampler[static_cast<std::size_t>(set)];
  std::vector<int> &occ = occupancy[static_cast<std::size_t>(set)];

  // Only reset a bucket the first time the timer enters it,
  // not on every access.
  uint64_t &last_bucket = last_occ_bucket[static_cast<std::size_t>(set)];
  if (current_bucket != last_bucket) {
    occ[static_cast<std::size_t>(current_bucket)] = 0;
    last_bucket = current_bucket;
  }

  // Check if this address was seen before in the sampled set
  for (sampler_entry &e : samp) {
    if (e.valid && e.tag == addr) {

      // A reuse distance beyond OPTgen's history window would alias around
      // the occupancy ring and look short: OPT could not have kept the line
      // that long, so treat it as an OPT miss.
      if (timer - e.last_time >= static_cast<uint64_t>(OCC_SIZE) * TIME_QUANTUM) {
        train(e.pc, false);
        e.pc = pc;
        e.last_time = timer;
        return;
      }

      uint64_t old_bucket = (e.last_time / TIME_QUANTUM) % OCC_SIZE;
      bool opt_hit = true;

      // Check if any bucket between the last time this address was seen and now
      // has occupancy >= the number of ways in the set. If so, this is not an OPT hit.
      uint64_t b = old_bucket;
      while (b != current_bucket) {
        if (occ[static_cast<std::size_t>(b)] >=
            static_cast<int>(sets[static_cast<std::size_t>(set)].lines.size())) {
          opt_hit = false;
          break;
        }
        b = (b + 1) % OCC_SIZE;
      }

      // Update the occupancy vector for all buckets between the last time this address was seen and now, if this is an OPT hit.
      if (opt_hit) {
        b = old_bucket;
        while (b != current_bucket) {
          if (occ[static_cast<std::size_t>(b)] <
              static_cast<int>(sets[static_cast<std::size_t>(set)].lines.size())) {
            occ[static_cast<std::size_t>(b)]++;
          }
          b = (b + 1) % OCC_SIZE;
        }
      }

      // Train the predictor based on whether this was an OPT hit or not
      train(e.pc, opt_hit);

      e.pc = pc;
      e.last_time = timer;

      return;
    }
  }

  // use empty sampler entry if available
  for (sampler_entry &e : samp) {
    if (!e.valid) {
      e.valid = true;
      e.tag = addr;
      e.pc = pc;
      e.last_time = timer;
      return;
    }
  }

  // Sampler full- evict LRU sampled entry
  auto victim = std::min_element(samp.begin(), samp.end(),
                                 [](const sampler_entry &a, const sampler_entry &b) {
                                   return a.last_time < b.last_time;
                                 });

  train(victim->pc, false);

  victim->valid = true;
  victim->tag = addr;
  victim->pc = pc;
  victim->last_time = timer;
}

// ---------------- MAIN POLICY ----------------

long hawkeye::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set,
                          const champsim::cache_block *current_set,
                          champsim::address ip, champsim::address full_addr,
                          access_type type) {
  return sets[static_cast<std::size_t>(set)].victim();
}

void hawkeye::replacement_cache_fill(uint32_t triggering_cpu, long set, long way,
                                     champsim::address full_addr,
                                     champsim::address ip,
                                     champsim::address victim_addr,
                                     access_type type) {
  uint64_t pc  = ip.to<uint64_t>();
  uint64_t addr = full_addr.to<uint64_t>();

  hawkeye_line &line =
      sets[static_cast<std::size_t>(set)].lines[static_cast<std::size_t>(way)];

  // Detrain the evicted line's PC if it was cache-friendly and
  // is still tracked in the sampler.
  if (line.valid && line.rrpv < MAX_RRIP && is_sampled_set(set)) {
    for (const sampler_entry &e : sampler[static_cast<std::size_t>(set)]) {
      if (e.valid && e.tag == line.tag) {
        train(line.pc, false);
        break;
      }
    }
  }

  // Writeback fills carry no PC (their ip is empty): skip all PC-based
  // training and prediction, and install the line as cache-averse.
  if (type == access_type::WRITE) {
    line.valid = true;
    line.tag   = addr;
    line.pc    = pc;
    line.rrpv  = MAX_RRIP;
    return;
  }

  // Train OPTgen on this fill access
  optgen_access(set, addr, pc);

  bool friendly = predict(pc);

  // Install new line
  line.valid   = true;
  line.tag     = addr;
  line.pc      = pc;

  if (friendly) {
    // Age cache-friendly lines, but only while none is saturated, so the
    // relative age order among friendly lines is preserved.
    bool saturated = false;
    for (const hawkeye_line &l : sets[static_cast<std::size_t>(set)].lines) {
      if (l.valid && l.rrpv == MAX_RRIP-1) {
        saturated = true;
        break;
      }
    }
    if (!saturated) {
      for (hawkeye_line &l : sets[static_cast<std::size_t>(set)].lines) {
        if (l.valid && l.rrpv < MAX_RRIP-1) {
          l.rrpv++;
        }
      }
    }
    line.rrpv = 0;
  } else {
    line.rrpv = MAX_RRIP;
  }
}

// update_replacement_state (on hits)
void hawkeye::update_replacement_state(uint32_t triggering_cpu, long set,
                                       long way, champsim::address full_addr,
                                       champsim::address ip,
                                       champsim::address victim_addr,
                                       access_type type, uint8_t hit) {
  // Consider only hits for training the predictor.
  // Hawkeye predictor uses PC to predict whether a line is cache-friendly or cache-averse.
  // If the access is a writeback, we don't have a PC to train the predictor, so we skip training in that case.
  if (!hit || access_type{type} == access_type::WRITE)
    return;

  uint64_t pc   = ip.to<uint64_t>();
  uint64_t addr = full_addr.to<uint64_t>();

  optgen_access(set, addr, pc);

  hawkeye_line &line =
      sets[static_cast<std::size_t>(set)].lines[static_cast<std::size_t>(way)];

  bool friendly = predict(pc);

  if (friendly) {
    line.rrpv = 0;
  } else {
    line.rrpv = MAX_RRIP;
  }
  line.pc = pc;
}