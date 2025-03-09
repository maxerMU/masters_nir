from dataclasses import dataclass, fields, asdict
import re

@dataclass
class Page:
    # buffer: int
    rel_id: int
    # is_local_temp: int
    fork_num: int
    block_num: int
    # mode: int
    relam: int
    relfilenode: int
    relhasindex: int
    # relpersistence: int
    relkind: int
    relnatts: int
    relfrozenxid: int
    relminmxid: int
    hit: int 

    def get_page_id(self) -> str:
        return f"{self.rel_id} {self.fork_num} {self.block_num}"

def read_pages(filepath: str):
    with open(filepath, "r") as file:
        data = file.read()

    pattern = r"buffer={(\d+)} rel_id={(\d+)} is_local_temp={(\w+)} fork_num={(\w+)} block_num={(\d+)} mode={(\w+)} strategy={} relam={(\d+)} relfilenode={(\d+)} relhasindex={(\w+)} relpersistence={(\w+)} relkind={(\w+)} relnatts={(\d+)} relfrozenxid={(\d+)} relminmxid={(\d+)} hit={(\w+)}"
    matches = re.findall(pattern, data)

    pages = []

    for match in matches:
        buffer = int(match[0])
        rel_id = int(match[1])
        is_local_temp = 1 if match[2] == "true" else 0
        fork_num = ["MAIN_FORKNUM", "FSM_FORKNUM", "VISIBILITYMAP_FORKNUM", "INIT_FORKNUM"].index(match[3])
        block_num = int(match[4])
        mode = ["RBM_NORMAL", "RBM_ZERO_AND_LOCK", "RBM_ZERO_AND_CLEANUP_LOCK", "RBM_ZERO_ON_ERROR", "RBM_NORMAL_NO_LOG"].index(match[5])
        relam = int(match[6])
        relfilenode = int(match[7])
        relhasindex = 1 if match[8] == "true" else 0
        relpersistence = ["p", "u", "t"].index(match[9])
        relkind = ["r", "i", "S", "t", "v", "m", "c", "f", "p", "I"].index(match[10])
        relnatts = int(match[11])
        relfrozenxid = int(match[12])
        relminmxid = int(match[13])
        hit = 1 if match[14] == "true" else 0

        page = Page(
            # buffer,
            rel_id,
            # is_local_temp, 
            fork_num,
            block_num,
            #mode,
            relam,
            relfilenode,
            relhasindex,
            # relpersistence,
            relkind,
            relnatts,
            relfrozenxid,
            relminmxid,
            hit,
        )
        pages.append(page)

    return pages

def save_pages_accs(filepath: str, pages: list[Page]):
    with open(filepath, "w") as f:
        for page in pages:
            f.write(f"{page.get_page_id()}\n")

def read_optimal_results(filepath: str):
    optimal_results = []
    with open(filepath, "r") as f:
        line = f.readline()
        while line:
            victims_count = int(line)
            victims_rates = []
            for _ in range(victims_count):
                victim_rate = f.readline().strip().split()
                victims_rates.append([int(victim_rate[0]), int(victim_rate[1])])

            optimal_results.append(victims_rates)
            line = f.readline()
    
    return optimal_results
