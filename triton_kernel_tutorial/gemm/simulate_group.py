import math

def cdiv(a, b):
    return (a + b - 1) // b


def simulate_pid_mapping(
    M, N, K,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    GROUP_SIZE_M
):
    num_pid_m = cdiv(M, BLOCK_SIZE_M)
    num_pid_n = cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_pid = num_pid_m * num_pid_n

    print(f"num_pid_m = {num_pid_m}, num_pid_n = {num_pid_n}")
    print(f"num_pid_in_group = {num_pid_in_group}")
    print("-" * 60)

    results = []

    for pid in range(total_pid):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        # Col-id of the program in the *launch grid*
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        results.append((pid, pid_m, pid_n))

        print(
            f"pid={pid:2d} | group={group_id} | "
            f"(pid_m, pid_n)=({pid_m}, {pid_n})"
        )

    return results

def simulate_pid_origin_mapping(
    M, N, K,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
):  
    num_pid_m = cdiv(M, BLOCK_SIZE_M)
    num_pid_n = cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_pid = num_pid_m * num_pid_n

    print(f"num_pid_m = {num_pid_m}, num_pid_n = {num_pid_n}")

    print("-" * 60)

    results = []

    for pid in range(total_pid):

        pid_n = pid % num_pid_n
        pid_m = pid // num_pid_n
        
        results.append((pid, pid_m, pid_n))

        print(
            f"pid={pid:2d} | "
            f"(pid_m, pid_n)=({pid_m}, {pid_n})"
        )

    results = []
    return results
if __name__ == "__main__":
    M, N, K = 9, 9 , 9
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 1,1,1
    GROUP_SIZE_M = 3
    print("origin padding")
    simulate_pid_origin_mapping(M,N,K,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)

    print("grouped padding")
    simulate_pid_mapping(M,N,K,BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,GROUP_SIZE_M)

"""

> python3 show_group.py
num_pid_m = 9, num_pid_n = 9
num_pid_in_group = 27
------------------------------------------------------------
pid= 0 | group=0 | (pid_m, pid_n)=(0, 0)
pid= 1 | group=0 | (pid_m, pid_n)=(1, 0)
pid= 2 | group=0 | (pid_m, pid_n)=(2, 0)
pid= 3 | group=0 | (pid_m, pid_n)=(0, 1)
pid= 4 | group=0 | (pid_m, pid_n)=(1, 1)
pid= 5 | group=0 | (pid_m, pid_n)=(2, 1)
pid= 6 | group=0 | (pid_m, pid_n)=(0, 2)
pid= 7 | group=0 | (pid_m, pid_n)=(1, 2)
pid= 8 | group=0 | (pid_m, pid_n)=(2, 2)
pid= 9 | group=0 | (pid_m, pid_n)=(0, 3)
pid=10 | group=0 | (pid_m, pid_n)=(1, 3)
pid=11 | group=0 | (pid_m, pid_n)=(2, 3)
pid=12 | group=0 | (pid_m, pid_n)=(0, 4)
pid=13 | group=0 | (pid_m, pid_n)=(1, 4)
pid=14 | group=0 | (pid_m, pid_n)=(2, 4)
pid=15 | group=0 | (pid_m, pid_n)=(0, 5)
pid=16 | group=0 | (pid_m, pid_n)=(1, 5)
pid=17 | group=0 | (pid_m, pid_n)=(2, 5)
pid=18 | group=0 | (pid_m, pid_n)=(0, 6)
pid=19 | group=0 | (pid_m, pid_n)=(1, 6)
pid=20 | group=0 | (pid_m, pid_n)=(2, 6)
pid=21 | group=0 | (pid_m, pid_n)=(0, 7)
pid=22 | group=0 | (pid_m, pid_n)=(1, 7)
pid=23 | group=0 | (pid_m, pid_n)=(2, 7)
pid=24 | group=0 | (pid_m, pid_n)=(0, 8)
pid=25 | group=0 | (pid_m, pid_n)=(1, 8)
pid=26 | group=0 | (pid_m, pid_n)=(2, 8)
pid=27 | group=1 | (pid_m, pid_n)=(3, 0)
pid=28 | group=1 | (pid_m, pid_n)=(4, 0)
pid=29 | group=1 | (pid_m, pid_n)=(5, 0)
pid=30 | group=1 | (pid_m, pid_n)=(3, 1)
pid=31 | group=1 | (pid_m, pid_n)=(4, 1)
pid=32 | group=1 | (pid_m, pid_n)=(5, 1)
pid=33 | group=1 | (pid_m, pid_n)=(3, 2)
pid=34 | group=1 | (pid_m, pid_n)=(4, 2)
pid=35 | group=1 | (pid_m, pid_n)=(5, 2)
pid=36 | group=1 | (pid_m, pid_n)=(3, 3)
pid=37 | group=1 | (pid_m, pid_n)=(4, 3)
pid=38 | group=1 | (pid_m, pid_n)=(5, 3)
pid=39 | group=1 | (pid_m, pid_n)=(3, 4)
pid=40 | group=1 | (pid_m, pid_n)=(4, 4)
pid=41 | group=1 | (pid_m, pid_n)=(5, 4)
pid=42 | group=1 | (pid_m, pid_n)=(3, 5)
pid=43 | group=1 | (pid_m, pid_n)=(4, 5)
pid=44 | group=1 | (pid_m, pid_n)=(5, 5)
pid=45 | group=1 | (pid_m, pid_n)=(3, 6)
pid=46 | group=1 | (pid_m, pid_n)=(4, 6)
pid=47 | group=1 | (pid_m, pid_n)=(5, 6)
pid=48 | group=1 | (pid_m, pid_n)=(3, 7)
pid=49 | group=1 | (pid_m, pid_n)=(4, 7)
pid=50 | group=1 | (pid_m, pid_n)=(5, 7)
pid=51 | group=1 | (pid_m, pid_n)=(3, 8)
pid=52 | group=1 | (pid_m, pid_n)=(4, 8)
pid=53 | group=1 | (pid_m, pid_n)=(5, 8)
pid=54 | group=2 | (pid_m, pid_n)=(6, 0)
pid=55 | group=2 | (pid_m, pid_n)=(7, 0)
pid=56 | group=2 | (pid_m, pid_n)=(8, 0)
pid=57 | group=2 | (pid_m, pid_n)=(6, 1)
pid=58 | group=2 | (pid_m, pid_n)=(7, 1)
pid=59 | group=2 | (pid_m, pid_n)=(8, 1)
pid=60 | group=2 | (pid_m, pid_n)=(6, 2)
pid=61 | group=2 | (pid_m, pid_n)=(7, 2)
pid=62 | group=2 | (pid_m, pid_n)=(8, 2)
pid=63 | group=2 | (pid_m, pid_n)=(6, 3)
pid=64 | group=2 | (pid_m, pid_n)=(7, 3)
pid=65 | group=2 | (pid_m, pid_n)=(8, 3)
pid=66 | group=2 | (pid_m, pid_n)=(6, 4)
pid=67 | group=2 | (pid_m, pid_n)=(7, 4)
pid=68 | group=2 | (pid_m, pid_n)=(8, 4)
pid=69 | group=2 | (pid_m, pid_n)=(6, 5)
pid=70 | group=2 | (pid_m, pid_n)=(7, 5)
pid=71 | group=2 | (pid_m, pid_n)=(8, 5)
pid=72 | group=2 | (pid_m, pid_n)=(6, 6)
pid=73 | group=2 | (pid_m, pid_n)=(7, 6)
pid=74 | group=2 | (pid_m, pid_n)=(8, 6)
pid=75 | group=2 | (pid_m, pid_n)=(6, 7)
pid=76 | group=2 | (pid_m, pid_n)=(7, 7)
pid=77 | group=2 | (pid_m, pid_n)=(8, 7)
pid=78 | group=2 | (pid_m, pid_n)=(6, 8)
pid=79 | group=2 | (pid_m, pid_n)=(7, 8)
pid=80 | group=2 | (pid_m, pid_n)=(8, 8)
root@hjbog-srdc-52:~/workspace/pyhip/leetGPU/2_maxtrix_multiplication#

"""