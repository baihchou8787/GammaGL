#!/usr/bin/env bash
# Run one GraphTokenizer task on a GPU in NVIDIA EXCLUSIVE_PROCESS mode.
set -euo pipefail

usage() {
    cat <<'EOF'
用法：run_exclusive_gpu.sh TASK_NAME OUTPUT_DIR COMMAND [ARG ...]

等待一张没有 CUDA 计算进程的卡，将其设为 EXCLUSIVE_PROCESS 后运行 COMMAND。
训练退出（成功、失败或收到中断）时恢复 Default 计算模式。

需要 sudo 免密授权 nvidia-smi 的 -c EXCLUSIVE_PROCESS 与 -c DEFAULT 操作。
可通过 GPU_IDS="0 1 2 3 4 5" 和 GPU_POLL_SECONDS=60 调整候选卡与轮询间隔。
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 3 ]]; then
    usage >&2
    exit 64
fi

task_name=$1
output_dir=$2
shift 2
gpu_ids=${GPU_IDS:-"0 1 2 3 4 5"}
poll_seconds=${GPU_POLL_SECONDS:-60}
nvidia_smi=${NVIDIA_SMI:-$(command -v nvidia-smi)}
active_gpu=""

mkdir -p "$output_dir"
log() {
    printf '%s task=%s %s\n' "$(date -Is)" "$task_name" "$*" >> "$output_dir/train.log"
}

cleanup() {
    if [[ -n $active_gpu ]]; then
        if sudo -n "$nvidia_smi" --id="$active_gpu" -c DEFAULT >/dev/null 2>&1; then
            log "state=released physical_gpu=$active_gpu compute_mode=Default"
        else
            log "state=release_failed physical_gpu=$active_gpu action=restore_Default"
        fi
        active_gpu=""
    fi
}
trap cleanup EXIT
trap 'exit 143' INT TERM

has_compute_processes() {
    "$nvidia_smi" --id="$1" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/{found=1} END{exit !found}'
}

attempt_gpu() {
    local gpu_id=$1 lock_file lock_fd status
    lock_file="/tmp/gammagl-current-gpu-${gpu_id}.lock"
    exec {lock_fd}>"$lock_file"
    release_lock() {
        flock -u "$lock_fd" || true
        exec {lock_fd}>&-
    }
    flock -n "$lock_fd" || { release_lock; return 75; }

    has_compute_processes "$gpu_id" && { release_lock; return 75; }
    if ! sudo -n "$nvidia_smi" --id="$gpu_id" -c EXCLUSIVE_PROCESS >/dev/null 2>&1; then
        log "state=exclusive_mode_failed physical_gpu=$gpu_id action=run_with_sudo"
        release_lock
        return 2
    fi
    if has_compute_processes "$gpu_id"; then
        sudo -n "$nvidia_smi" --id="$gpu_id" -c DEFAULT >/dev/null 2>&1 || true
        release_lock
        return 75
    fi

    active_gpu=$gpu_id
    log "state=claimed physical_gpu=$gpu_id compute_mode=EXCLUSIVE_PROCESS"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@" >> "$output_dir/train.log" 2>&1
    status=$?
    set -e
    cleanup
    release_lock
    return "$status"
}

log "state=queued gpu_range=$gpu_ids exclusivity=EXCLUSIVE_PROCESS"
while true; do
    for gpu_id in $gpu_ids; do
        if attempt_gpu "$gpu_id" "$@"; then
            exit 0
        fi
        status=$?
        if [[ $status -ne 75 ]]; then
            exit "$status"
        fi
    done
    log "state=waiting_for_any_idle_gpu"
    sleep "$poll_seconds"
done
