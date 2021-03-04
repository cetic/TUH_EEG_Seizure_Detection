# Job scripts

The job scripts are used with Slurm to start a job. But first you can use the interactive mode.

## Slurm interactive mode

```bash
$srun --partition=gpu --job-name=xgb -N 1 -n 16 --mem=120G\
-t 1:00:00 --gres="gpu:3" --pty bash
```

This command will start a job (`bash`) in interactive mode (option `--pty`). We ask to put the job on the `gpu` partition (`--partition=gpu`) and to use 3 GPUs (`--gres="gpu:3"`). The job is called `xgb` (`--job-name=xgb`), will run on 1 node (`-N 1`) with 16 cores (`-n 16`), 120 GB of RAM (`--mem=120G`) during maximum 1 hour (`-t 1:00:00`)

N.B.: we could use the `debug` partition but we would have to use less RAM and no GPU.

## Slurm, sbatch

The following script can be run with `sbatch run_toto.sh`

```bash
#!/bin/bash
#SBATCH --job-name=test_sbash
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=5G
#SBATCH --gres="gpu:2"
#SBATCH -t 1:00:00
#SBATCH --mail-user=vincent.stragier@student.mumons.ac.be
#SBATCH --mail-type=ALL

# Loading Python 3.8.6rc1
module use "$HOME"/modulefiles/
module load python/3.8.6rc1

# Display the GPU information
echo "DATE : $(date)"
echo "_____________________________________________"
echo " HOSTNAME             : $HOSTNAME"
echo "_____________________________________________"
echo " CUDA_DEVICE_ORDER    : $CUDA_DEVICE_ORDER"
echo "_____________________________________________"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "_____________________________________________"
nvidia-smi -L
echo "_____________________________________________"

# Starting the Python program and printing the time it took to complete
time python -V

# Returns the size of the following folder
# du -sh ~/TUH_SZ_v1.5.2/TUH/
# du -sh ~/CHBMIT/

# Returns the amount of free space
free -h

# Returns the CPU informations
# cat /proc/cpuinfo
lscpu
```

### Usage example

```bash
stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection/jobs_scripts$ sbatch run_toto.sh
Submitted batch job 6495
stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection/jobs_scripts$ ls
run_toto.sh  slurm-6495.out
stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection/jobs_scripts$ cat slurm-6495.out
DATE : mardi 24 novembre 2020, 13:23:26 (UTC+0100)
_____________________________________________
 HOSTNAME             : simu1
_____________________________________________
 CUDA_DEVICE_ORDER    :
_____________________________________________
 CUDA_VISIBLE_DEVICES : 0,1
_____________________________________________
GPU 0: GeForce GTX 1080 Ti (UUID: GPU-0292f655-b03a-bc37-6538-b66d01c46a58)
GPU 1: GeForce GTX 1080 Ti (UUID: GPU-331add13-8886-3c2e-777d-bbef47b8bd3a)
_____________________________________________
Python 3.8.6rc1

real    0m0.120s
user    0m0.000s
sys     0m0.000s
              total        used        free      shared  buff/cache   available
Mem:           125G        900M        4,4G        1,1G        120G        123G
Swap:          127G        156M        127G
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                32
On-line CPU(s) list:   0-31
Thread(s) per core:    2
Core(s) per socket:    8
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 79
Model name:            Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
Stepping:              1
CPU MHz:               1200.199
CPU max MHz:           3000,0000
CPU min MHz:           1200,0000
BogoMIPS:              4201.30
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              20480K
NUMA node0 CPU(s):     0-7,16-23
NUMA node1 CPU(s):     8-15,24-31
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer
aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb invpcid_single intel_pt ssbd ibrs ibpb stibp kaiser tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdseed adx smap xsaveopt cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts md_clear flush_l1d
```

## Sources

- [UMONS cluster documentation](https://cluster.ig.umons.ac.be/docs/en/0.1/slurm.html)
- [CÉCI documentation](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html#creating-a-job)
