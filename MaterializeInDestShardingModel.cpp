/// Sharding model for bufferization.materialize_in_destination.
///
/// This op is a *sink* at the tensor->memref boundary: it materializes a tensor
/// value into a destination buffer (memref). It does not represent computation
/// and does not introduce loop structure.
///
/// Design choices (conservative):
/// - No loop/reduction information is exposed (empty iterator lists).
/// - Sharding is only relevant for the tensor operand (typically operand 0).
///   Destination memrefs are not annotated with shard information in this design.
/// - We do not materialize (insert) shard annotations at this sink. Shard
///   annotations are expected to live on tensor SSA values earlier in the pipeline.
/// - Partitioning is implemented by cloning the op with already-partitioned
///   operands; we do not generate additional slicing/tiling here.
struct MaterializeInDestShardingModel
    : public shard::ShardingInterface::ExternalModel<
          MaterializeInDestShardingModel,
          bufferization::MaterializeInDestinationOp> {

  /// Sink/boundary op: no loop iterator information.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {};
  }

  /// Sink/boundary op: no reduction loop information.
  SmallVector<shard::ReductionKind>
  getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  /// No structured indexing maps for this op.
  SmallVector<AffineMap> getIndexingMaps(Operation *) const { return {}; }

  /// Choose a sharding option.
  ///
  /// The op typically has no results, so resultShardings are irrelevant here.
  /// We only care about the tensor operand sharding (usually operand 0).
  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    (void)resultShardings; // intentionally unused (no results)

    if (op->getNumOperands() < 1)
      return failure();

    // Preserve the tensor operand sharding if provided.
    if (operandShardings.size() >= 1 && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    // Otherwise, do not constrain sharding.
    return shard::ShardingOption::makeEmpty();
  }

  /// Return sharding annotations for operands/results.
  ///
  /// We return a sharding for operand 0 (tensor) and "no sharding" for the
  /// remaining operands (usually destination memrefs).
  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t rank = tensorTy ? tensorTy.getRank() : 0;

    shard::Sharding tensorSharding = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(op->getNumOperands() + op->getNumResults());

    // operand 0: tensor source
    res.push_back(tensorSharding);

    // remaining operands: destination memrefs / other => no sharding
    for (unsigned i = 1; i < op->getNumOperands(); ++i)
      res.push_back(shard::Sharding());

    return res;
  }

  /// Sink op: do not add (materialize) shard annotations here.
  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  /// Partition by cloning.
  ///
  /// The partitioner supplies already-partitioned operands; we just clone this
  /// sink op to keep semantics, without generating additional slicing/tiling.
  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding> operandShardings,
                          ArrayRef<shard::Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTableCollection,
                          OpBuilder &builder) const {
    (void)operandShardings;
    (void)resultShardings;
    (void)symbolTableCollection;
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

