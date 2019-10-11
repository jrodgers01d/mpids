def get_local_data(array_data, distribution, procs, rank):
        data_length = len(array_data)
        local_low_idx = low_block(data_length, procs, rank)
        local_high_idx = high_block(data_length, procs, rank)

        return array_data[local_low_idx: local_high_idx]

def low_block(length, procs, rank):
        return (length * rank) // procs

def high_block(length, procs, rank):
        return low_block(length, procs, rank + 1)
