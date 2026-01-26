/// Return the set of loop iterator indices used by an affine map result list.
///
/// We only consider direct AffineDimExpr results (no affine.apply / no arithmetic),
/// keeping this intentionally conservative.
static llvm::SmallDenseSet<int64_t>
collectLoopItersUsedByMap(AffineMap map) {
  llvm::SmallDenseSet<int64_t> used;
  for (AffineExpr e : map.getResults()) {
    if (auto d = dyn_cast<AffineDimExpr>(e))
      used.insert(d.getPosition());
  }
  return used;
}

/// Return true if the map is a pure projected permutation from loop dims to
/// tensor dims (i.e. results are distinct AffineDimExpr).
static bool isProjectedPermutationOfDims(AffineMap map) {
  // MLIR already provides a check.
  // This will be false if there is any arithmetic / symbols / repeated dims.
  return map.isProjectedPermutation();
}

/// Recognize the canonical 2D matmul contraction pattern:
///   iterators: [parallel, parallel, reduction]  (possibly with extra loops ignored)
///   maps: A(i,k), B(k,j), C(i,j)
///
/// This is meant to catch the common linalg.generic that came from matmul-like
/// lowering (or hand-written).
static bool matchMatmulLike(linalg::GenericOp op,
                            int64_t &iIter,
                            int64_t &jIter,
                            int64_t &kIter) {
  auto iters = op.getIteratorTypesArray();
  if (iters.size() < 3)
    return false;

  // Collect parallel and reduction iter indices.
  SmallVector<int64_t> parallelIters;
  SmallVector<int64_t> reductionIters;
  for (int64_t idx = 0; idx < (int64_t)iters.size(); ++idx) {
    if (iters[idx] == utils::IteratorType::parallel)
      parallelIters.push_back(idx);
    else if (iters[idx] == utils::IteratorType::reduction)
      reductionIters.push_back(idx);
  }

  // Canonical matmul has exactly 2 parallel + 1 reduction.
  // If the IR has extra loops, we relax this, but keep it conservative.
  if (parallelIters.size() != 2 || reductionIters.size() != 1)
    return false;

  // Must have at least 2 inputs + 1 output for matmul-like.
  if (op.getNumInputs() < 2 || op.getNumOutputs() < 1)
    return false;

  auto maps = op.getIndexingMapsArray();
  // Convention in linalg.generic: maps are [inputs..., outputs...]
  AffineMap aMap = maps[0];
  AffineMap bMap = maps[1];
  AffineMap cMap = maps[op.getNumInputs() + 0];

  // Ensure maps are simple permutations/projections.
  if (!isProjectedPermutationOfDims(aMap) ||
      !isProjectedPermutationOfDims(bMap) ||
      !isProjectedPermutationOfDims(cMap))
    return false;

  // Collect iter usage sets.
  auto aUsed = collectLoopItersUsedByMap(aMap);
  auto bUsed = collectLoopItersUsedByMap(bMap);
  auto cUsed = collectLoopItersUsedByMap(cMap);

  // For canonical A(i,k), B(k,j), C(i,j):
  // - C uses both parallel iters and *not* the reduction iter.
  // - A uses i and k
  // - B uses k and j
  int64_t p0 = parallelIters[0];
  int64_t p1 = parallelIters[1];
  int64_t r0 = reductionIters[0];

  // C must use both parallel and must not use reduction.
  if (!(cUsed.contains(p0) && cUsed.contains(p1)))
    return false;
  if (cUsed.contains(r0))
    return false;

  // A must use (one parallel) + (reduction)
  // B must use (other parallel) + (reduction)
  bool aIs_p0_r = aUsed.contains(p0) && aUsed.contains(r0) && !aUsed.contains(p1);
  bool aIs_p1_r = aUsed.contains(p1) && aUsed.contains(r0) && !aUsed.contains(p0);
  bool bIs_p0_r = bUsed.contains(p0) && bUsed.contains(r0) && !bUsed.contains(p1);
  bool bIs_p1_r = bUsed.contains(p1) && bUsed.contains(r0) && !bUsed.contains(p0);

  if (aIs_p0_r && bIs_p1_r) {
    iIter = p0;
    jIter = p1;
    kIter = r0;
    return true;
  }
  if (aIs_p1_r && bIs_p0_r) {
    iIter = p1;
    jIter = p0;
    kIter = r0;
    return true;
  }

  return false;
}

static bool shardingIntroducesGlobalReduction(linalg::GenericOp op,
                                             int64_t shardIter) {
  // ---------------------------------------------------------------------------
  // Scalar result heuristic: scalar reductions are typically global.
  // ---------------------------------------------------------------------------
  if (op.getNumResults() == 1) {
    if (auto rt = dyn_cast<RankedTensorType>(op.getResultTypes()[0])) {
      if (rt.getRank() == 0)
        return true;
    }
  }

  // ---------------------------------------------------------------------------
  // Fast-path: recognize matmul-like contraction.
  // Sharding on i or j (the parallel iters that index C) is safe (no all-reduce).
  // Sharding on k (the reduction iter) would require reduction across shards.
  // ---------------------------------------------------------------------------
  int64_t iIter = -1, jIter = -1, kIter = -1;
  if (matchMatmulLike(op, iIter, jIter, kIter)) {
    if (shardIter == iIter || shardIter == jIter)
      return false;
    if (shardIter == kIter)
      return true;
    // If shardIter is something else (shouldn't happen in strict match), be conservative.
    return true;
  }

  // ---------------------------------------------------------------------------
  // Conservative fallback:
  //
  // If shardIter is a reduction iterator, it often implies each shard computes a
  // partial reduction that must be combined globally.
  //
  // We make it slightly less trigger-happy by checking whether the output indexing
  // map depends on shardIter. If the output depends on shardIter, then sharding that
  // iter changes which output elements are produced by each shard (more like
  // partitioning the output) and may not require all-reduce.
  // ---------------------------------------------------------------------------
  auto iters = op.getIteratorTypesArray();
  if (shardIter >= 0 && shardIter < (int64_t)iters.size() &&
      iters[shardIter] == utils::IteratorType::reduction) {
    // Look at the first output map (common case).
    auto maps = op.getIndexingMapsArray();
    if (op.getNumOutputs() > 0) {
      AffineMap outMap = maps[op.getNumInputs() + 0];
      auto outUsed = collectLoopItersUsedByMap(outMap);
      // If the output *does not* depend on the sharded reduction iter,
      // then shards are computing partial sums for the same output -> needs all-reduce.
      if (!outUsed.contains(shardIter))
        return true;
    }
    // Otherwise, keep conservative: still likely needs all-reduce.
    return true;
  }

  // ---------------------------------------------------------------------------
  // Another conservative heuristic:
  // If the op has any reduction iterator at all, and you shard a parallel iterator
  // that does NOT appear in the output map, you are probably partitioning a reduced
  // dimension (i.e. different shards reduce different slices but to the same output),
  // which requires all-reduce.
  // ---------------------------------------------------------------------------
  bool hasReduction = llvm::any_of(iters, [](utils::IteratorType t) {
    return t == utils::IteratorType::reduction;
  });

  if (hasReduction && op.getNumOutputs() > 0) {
    auto maps = op.getIndexingMapsArray();
    AffineMap outMap = maps[op.getNumInputs() + 0];
    auto outUsed = collectLoopItersUsedByMap(outMap);

    if (!outUsed.contains(shardIter)) {
      // Sharding a loop that does not index the output while reductions exist
      // strongly suggests partial reductions that must be combined.
      return true;
    }
  }

  // ---------------------------------------------------------------------------
  // Default: assume no global reduction is required.
  // This keeps the planner from inserting collectives too eagerly.
  // ---------------------------------------------------------------------------
  return false;
}

