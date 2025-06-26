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

