import time

def rank_chunks_by_ratings(retrieved_chunks, chunk_ratings):
    for chunk in retrieved_chunks:
        if chunk in chunk_ratings:
            chunk_ratings[chunk] += 1
        else:
            chunk_ratings[chunk] = 1
    return chunk_ratings

def run_with_timer(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    
    return elapsed, result

