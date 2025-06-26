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
