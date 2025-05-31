Buffer
ReadBufferExtended(Relation reln, ForkNumber forkNum, BlockNumber blockNum,
				   ReadBufferMode mode, BufferAccessStrategy strategy)
{
	bool		hit;
	Buffer		buf;

	if (RELATION_IS_OTHER_TEMP(reln))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cannot access temporary tables of other sessions")));

	pgstat_count_buffer_read(reln);
	buf = ReadBuffer_common(RelationGetSmgr(reln), reln->rd_rel->relpersistence,
							forkNum, blockNum, mode, strategy, &hit);

	const char *forkNumStr = "";
	if (forkNum == MAIN_FORKNUM)
		forkNumStr = "MAIN_FORKNUM";
	else if (forkNum == FSM_FORKNUM)
		forkNumStr = "FSM_FORKNUM";
	else if (forkNum == VISIBILITYMAP_FORKNUM)
		forkNumStr = "VISIBILITYMAP_FORKNUM";
	else if (forkNum == INIT_FORKNUM)
		forkNumStr = "INIT_FORKNUM";

	const char *readBufModeStr = "";
	if (mode == RBM_NORMAL)
		readBufModeStr = "RBM_NORMAL";
	else if (mode == RBM_ZERO_AND_LOCK)
		readBufModeStr = "RBM_ZERO_AND_LOCK";
	else if (mode == RBM_ZERO_AND_CLEANUP_LOCK)
		readBufModeStr = "RBM_ZERO_AND_CLEANUP_LOCK";
	else if (mode == RBM_ZERO_ON_ERROR)
		readBufModeStr = "RBM_ZERO_ON_ERROR";
	else if (mode == RBM_NORMAL_NO_LOG)
		readBufModeStr = "RBM_NORMAL_NO_LOG";

	elog(WARNING, "\n=======================================\nbuffer={%d} rel_id={%u} is_local_temp={%s} fork_num={%s} block_num={%u} mode={%s} strategy={} relam={%u} relfilenode={%u} relhasindex={%s} relpersistence={%c} relkind={%c} relnatts={%d} relfrozenxid={%u} relminmxid={%u} hit={%s}\n=======================================\n",
		 buf,
		 reln->rd_rel->oid,
		 SmgrIsTemp(RelationGetSmgr(reln)) ? "true" : "false",
		 forkNumStr,
		 blockNum,
		 readBufModeStr,
		 // strategyStr,
		 reln->rd_rel->relam,
		 reln->rd_rel->relfilenode,
		 reln->rd_rel->relhasindex ? "true" : "false",
		 reln->rd_rel->relpersistence,
		 reln->rd_rel->relkind,
		 reln->rd_rel->relnatts,
		 reln->rd_rel->relfrozenxid,
		 reln->rd_rel->relminmxid,
		 hit ? "true" : "false");

	if (hit)
		pgstat_count_buffer_hit(reln);
	return buf;
}
