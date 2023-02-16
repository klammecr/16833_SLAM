##CODE 1
# from multiprocessing import Process

# def print_func(continent = 'Asia'):
#     print('Name of continent is : ', continent)

# if __name__ == "__main__":
#     names = ['North America', 'India', 'Europe']
#     procs = []
#     proc = Process(target=print_func)
#     procs.append(proc)
#     proc.start()

#     for name in names:
#         proc = Process(target=print_func, args=(name,))
#         procs.append(proc)
#         proc.start()
        
#     for proc in procs:
#         proc.join()

##CODE 1
# from multiprocessing import Lock, Process, Queue, current_process
# import time
# import queue

# def do_job(tasks_to_accomplish, tasks_that_are_done):
#     while True:
#         try:
#             task = tasks_to_accomplish.get_nowait()
#         except queue.Empty:
#             break
#         else:
#             print(task)
#             tasks_that_are_done.put(task + ' is done by ' + current_process().name)
#             time.sleep(0.5)
    
#     return True

# def main():
#     number_of_task = 10
#     number_of_processes = 4
#     tasks_to_accomplish = Queue()
#     tasks_that_are_done = Queue()
    
#     processes = []
    
#     for i in range(number_of_task):
#         tasks_to_accomplish.put("Task no " + str(i))
    
#     for w in range(number_of_processes):
#         p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
#         processes.append(p)
#         p.start()
        
#     for p in processes:
#         p.join()
        
#     while not tasks_that_are_done.empty():
#         print(tasks_that_are_done.get())
    
#     return True

# if __name__ == "__main__":
#     main()        

# #CODE 3
# from multiprocessing import Pool

# import time

# work = [["A", 5,], ["B", 2], ["C", 1], ["E", 4]]

# def work_log(work_data):
#     print("Process {} waiting for {} seconds".format(work_data[0], work_data[1]))
#     time.sleep(int(work_data[1]))
#     print("Process {} is finished".format(work_data[0]))

# def pool_handler():
#     p = Pool(2)
#     p.map(work_log, work)

# if __name__ == "__main__":
#     pool_handler()

#CODE 5
import numpy as np
import time
from multiprocessing import Pool

# Define a simple function to apply to each element of the array
def func(x):
    return x * x

if __name__ == "__main__":
    # Generate a large NumPy array
    arr_size = 10000000
    arr = np.arange(arr_size)

    # Define the number of processes to use with starmap()
    num_processes = 4

    # Measure the time taken to apply the function with starmap()
    start_time = time.time()
    with Pool(num_processes) as p:
        result = p.starmap(func, [(x,) for x in arr])
    end_time = time.time()

    print(f"Time taken with starmap: {end_time - start_time:.4f} seconds")

    # Measure the time taken to apply the function with iterations
    start_time = time.time()
    result = [func(x) for x in arr]
    end_time = time.time()

    print(f"Time taken with iterations: {end_time - start_time:.4f} seconds")


        