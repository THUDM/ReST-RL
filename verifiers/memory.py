# settings for memory usage in execution
# for example, you may use a maximum of 16G memory for each process for 1T memory in total
max_memory_as_bytes = 16 * 1024 * 1024 * 1024
print(f"Setting maximum memory usage per process to: {max_memory_as_bytes * 1.0 / 1024 / 1024 / 1024 if max_memory_as_bytes is not None else '--'} GB\n")
