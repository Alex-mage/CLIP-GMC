import os
import itertools
import subprocess
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="运行不同参数组合的实验")

    # 定义要遍历的参数及其可能的值
    parser.add_argument("--model_types", nargs="+", default=["c2", "c3"],
                        help="模型类型列表")
    parser.add_argument("--text_ps", nargs="+", default=["des3"],
                        help="文本提示类型列表")
    parser.add_argument("--epochs", nargs="+", type=int, default=[10])
    parser.add_argument("--data_ratio", nargs="+", type=float, default=[-1.0, 0.05],
                        help="训练数据比例列表")
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 47, 48],
                        help="随机种子列表")

    args = parser.parse_args()

    # 生成所有参数组合
    combinations = list(itertools.product(
        args.model_types,
        args.text_ps,
        args.seeds,
        args.data_ratio,
        args.epochs
    ))

    total_combinations = len(combinations)
    print(f"总共将运行 {total_combinations} 种参数组合")

    # 遍历所有组合并运行
    for i, (model_type, text_p, seed, data_ratio, epochs) in enumerate(combinations):
        # 构建命令
        cmd = f"torchrun --nproc_per_node=4 main.py "
        cmd += f"--model_type {model_type} "
        cmd += f"--text_p {text_p} "
        cmd += f"--seed {seed} "
        cmd += f"--data_ratio {data_ratio}"
        cmd += f" --epochs {epochs}"

        # 记录开始时间
        start_time = time.time()

        print(f"\n[{i + 1}/{total_combinations}] 运行组合: model_type={model_type}, text_p={text_p}, seed={seed}")
        print(f"命令: {cmd}")

        # 创建日志目录（如果需要）
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"model_{model_type}_text_{text_p}_seed_{seed}.log")

        # 运行命令
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # 实时输出日志
            with open(log_file, "w") as f:
                for line in process.stdout:
                    print(line, end="")
                    f.write(line)

            # 等待进程完成
            process.wait()

            # 检查返回码
            if process.returncode != 0:
                print(f"警告: 组合 {model_type}_{text_p}_{seed} 运行失败，返回码: {process.returncode}")

        except Exception as e:
            print(f"错误: 运行组合 {model_type}_{text_p}_{seed} 时发生异常: {str(e)}")

        # 记录结束时间和耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"组合 {model_type}_{text_p}_{seed} 运行完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

        # 在每个组合之间稍作暂停，以便系统资源释放
        time.sleep(5)
        # 只保留日志文件的最后十行
        with open(log_file, "r") as f:
            lines = f.readlines()
        with open(log_file, "w") as f:
            f.writelines(lines[-10:])

    print("\n所有参数组合运行完成!")


if __name__ == "__main__":
    main()
