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

# #CODE 5
# import numpy as np
# import time
# from multiprocessing import Pool

# # Define a simple function to apply to each element of the array
# def func(x):
#     return x * x

# if __name__ == "__main__":
#     # Generate a large NumPy array
#     arr_size = 10000000
#     arr = np.arange(arr_size)

#     # Define the number of processes to use with starmap()
#     num_processes = 4

#     # Measure the time taken to apply the function with starmap()
#     start_time = time.time()
#     with Pool(num_processes) as p:
#         result = p.starmap(func, [(x,) for x in arr])
#     end_time = time.time()

#     print(f"Time taken with starmap: {end_time - start_time:.4f} seconds")

#     # Measure the time taken to apply the function with iterations
#     start_time = time.time()
#     result = [func(x) for x in arr]
#     end_time = time.time()

#     print(f"Time taken with iterations: {end_time - start_time:.4f} seconds")


# #CODE 6
# from multiprocessing import Process, Queue
# import difflib, random, time

# def f2(wordlist, mainwordlist, q):
#     for mainword in mainwordlist:
#         matches = difflib.get_close_matches(mainword,wordlist,len(wordlist),0.7)
#         q.put(matches)

# if __name__ == '__main__':

#     # constants (for 50 input words, find closest match in list of 100 000 comparison words)
#     q = Queue()
#     wordlist = ["".join([random.choice([letter for letter in "abcdefghijklmnopqersty"]) for lengthofword in range(5)]) for nrofwords in range(100000)]
#     mainword = "hello"
#     mainwordlist = [mainword for each in range(50)]

#     # normal approach
#     t = time.time()
#     for mainword in mainwordlist:
#         matches = difflib.get_close_matches(mainword,wordlist,len(wordlist),0.7)
#         q.put(matches)
#     print("t1 = ", time.time()-t)

#     # split work into 5 or 10 processes
#     processes = 8
#     def splitlist(inlist, chunksize):
#         return [inlist[x:x+chunksize] for x in range(0, len(inlist), chunksize)]
#     print(len(mainwordlist)/processes)
#     mainwordlistsplitted = splitlist(mainwordlist, len(mainwordlist)//processes)
#     print("list ready")

#     t = time.time()
#     proc_list = []
#     for submainwordlist in mainwordlistsplitted:
#         print("sub")
#         p = Process(target=f2, args=(wordlist,submainwordlist,q,))
#         p.Daemon = True
#         proc_list.append(p)
#         p.start()
        
#     for p in proc_list:
#         p.join()
        
#     print("t2 = ", time.time()-t)
#     # while True:
#     #     print(q.get())

# #CODE 7
# from multiprocessing import Process, Queue
# import difflib, random, time
# import numpy as np

# def func(num):
#     # r = num*np.random.normal(0, 1, (1000, 1000))
#     # y = np.matmul(r, r)
#     time.sleep(0.005)
#     return True

# def f2(numlist, q):
#     for num in numlist:
#         y = func(num)
#         q.put(y)

# if __name__ == '__main__':

#     # constants (for 50 input words, find closest match in list of 100 000 comparison words)
#     q = Queue()
#     numlist = [nums for nums in range(1000)]

#     # normal approach
#     t = time.time()
#     for num in numlist:
#         out = func(num)
#         q.put(out)
        
#     print("t1 = ", time.time()-t)

#     # # split work into 5 or 10 processes
#     processes = 8
#     chunksize = len(numlist)//processes
#     numlist_split = [numlist[x:x+chunksize] for x in range(0, len(numlist), chunksize)]
#     print(len(numlist)/processes)
#     print("list ready")

#     t = time.time()
#     proc_list = []
#     for numlist_split_item in numlist_split:
#         print("sub")
#         p = Process(target=f2, args=(numlist_split_item,q,))
#         p.Daemon = True
#         proc_list.append(p)
#         p.start()
        
#     for p in proc_list:
#         p.join()
        
#     print("t2 = ", time.time()-t)
#     # while True:
#     #     print(q.get())

#CODE 8
from joblib import Parallel, delayed
from math import sqrt
import time

t = time.time()
Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10**7))
print("t1 = ", time.time()-t)

t = time.time()
Parallel(n_jobs=8)(delayed(sqrt)(i**2) for i in range(10**7))
print("t2 = ", time.time()-t)