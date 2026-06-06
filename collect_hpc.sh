folder_path="./rs_error/rs-0"
filenames=()
count=0
for filename in "$folder_path"/*
do
    if [ -f "$filename" ];then
        filenames[$count]="$filename"
        count=$((count + 1))
    fi
done

for str in ${filenames[@]}
do
OLD_IFS="$IFS"
IFS='/' 
arr=($str)
IFS="$OLD_IFS"
p2=${arr[2]}
echo $p2
# 快照恢复，延长等待时间确保恢复完全
VBoxManage snapshot vmwin7 restore 'snapfile0111'
sleep 8
# 启动虚拟机，延长等待时间确保系统稳定
VBoxManage startvm 'vmwin7'
sleep 30
# 同步Windows虚拟机时钟，确保时序一致
sshpass -p '123' ssh vm@192.168.56.102 w32tm /resync /nowait
sleep 2
# 获取虚拟机进程PID
pid_virbox=$(ps aux | grep "vmwin7" | awk 'NR==1{print $2}')
# SSH压缩传输勒索软件样本，减少I/O噪声
sshpass -p '123' scp -C $str vm@192.168.56.102:/D/Malshared
sleep 5  # 等待I/O恢复稳定
# 删除原有任务，避免残留干扰
sshpass -p '123' ssh vm@192.168.56.102 schtasks /delete /tn "MY_NOTEPAD" /f
sleep 2
# 创建新任务
sshpass -p '123' ssh vm@192.168.56.102 schtasks /create /tn "MY_NOTEPAD" /tr "D:\Malshared\/$p2" /sc once /st 00:00
sleep 3
# 启动任务，延迟0.5秒启动perf采样，规避启动波动
sshpass -p '123' ssh vm@192.168.56.102 schtasks /run /tn "MY_NOTEPAD"
sleep 0.5
# 优化perf采样参数，延长采样时长，完善监测指标
sudo perf stat -p $pid_virbox -I 100 -e branches,branch-misses,L1-icache-load-misses,iTLB-load-misses,instructions,cpu-clock,page-faults 2> /home/zhao/Documents/20250606_exe_all/rs_error/rs-0-result/$p2.csv sleep 2s
sleep 2
# 关闭虚拟机，避免残留
VBoxManage controlvm vmwin7 poweroff
done
# 实验结束后恢复快照，确保下次实验环境干净
VBoxManage snapshot vmwin7 restore 'snapfile0111'