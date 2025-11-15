#!/usr/bin/env python3
"""
Self-Organizing-Gaussians (SOG) 训练包装器
用于GSSE GUI集成
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml
from typing import Optional, Dict, Any


class SOGTrainer:
    """Self-Organizing-Gaussians训练器包装类"""
    
    def __init__(self, sog_path: Optional[str] = None):
        """
        初始化SOG训练器
        
        Args:
            sog_path: Self-Organizing-Gaussians的路径，默认使用依赖目录
        """
        if sog_path is None:
            # 默认使用依赖目录中的Self-Organizing-Gaussians
            gsse_dir = os.path.dirname(os.path.abspath(__file__))
            sog_path = os.path.join(gsse_dir, "dependencies", "Self-Organizing-Gaussians")
        
        self.sog_path = sog_path
        self.train_script = os.path.join(self.sog_path, "train.py")
        self.config_dir = os.path.join(self.sog_path, "config")
        
        # 验证路径
        if not os.path.exists(self.train_script):
            raise FileNotFoundError(f"SOG训练脚本不存在: {self.train_script}")
        if not os.path.exists(self.config_dir):
            raise FileNotFoundError(f"SOG配置目录不存在: {self.config_dir}")
    
    def create_custom_config(
        self,
        output_path: str,
        dataset_path: str,
        model_path: str,
        iterations: int = 30000,
        use_sh: bool = True,
        test_iterations: Optional[list] = None,
        save_iterations: Optional[list] = None,
        compress_iterations: Optional[list] = None,
        config_name: str = "custom_config"
    ) -> str:
        """
        创建自定义配置文件
        
        Args:
            output_path: 输出路径
            dataset_path: 数据集路径
            model_path: 模型保存路径
            iterations: 训练迭代次数
            use_sh: 是否使用球谐函数
            test_iterations: 测试迭代点
            save_iterations: 保存迭代点
            compress_iterations: 压缩迭代点
            config_name: 配置文件名
        
        Returns:
            配置文件路径
        """
        if test_iterations is None:
            test_iterations = [7000, 10000, 20000, 30000]
        if save_iterations is None:
            save_iterations = [7000, 10000, 20000, 30000]
        if compress_iterations is None:
            compress_iterations = [7000, 10000, 20000, 30000]
        
        # 读取默认配置作为模板
        default_config_path = os.path.join(self.config_dir, "ours_q_sh_local_test.yaml")
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新配置
        config['dataset']['source_path'] = dataset_path
        config['dataset']['model_path'] = model_path
        config['optimization']['iterations'] = iterations
        config['run']['use_sh'] = use_sh
        config['run']['test_iterations'] = test_iterations
        config['run']['save_iterations'] = save_iterations
        config['run']['compress_iterations'] = compress_iterations
        config['run']['no_progress_bar'] = False  # 在GUI中显示进度条
        
        # 保存自定义配置
        config_file_path = os.path.join(output_path, f"{config_name}.yaml")
        os.makedirs(output_path, exist_ok=True)
        
        with open(config_file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return config_file_path
    
    def train(
        self,
        dataset_path: str,
        output_path: str,
        iterations: int = 30000,
        config_name: str = "ours_q_sh_local_test",
        custom_config: Optional[str] = None,
        use_sh: bool = True,
        test_iterations: Optional[list] = None,
        save_iterations: Optional[list] = None,
        compress_iterations: Optional[list] = None,
        additional_args: Optional[Dict[str, Any]] = None
    ) -> subprocess.Popen:
        """
        启动SOG训练
        
        Args:
            dataset_path: 数据集路径
            output_path: 输出路径
            iterations: 训练迭代次数
            config_name: 配置文件名（不包括.yaml后缀）
            custom_config: 自定义配置文件路径，如果提供则忽略config_name
            use_sh: 是否使用球谐函数
            test_iterations: 测试迭代点
            save_iterations: 保存迭代点
            compress_iterations: 压缩迭代点
            additional_args: 额外的命令行参数
        
        Returns:
            训练进程对象
        """
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 设置默认迭代点
        if test_iterations is None:
            test_iterations = [7000, 10000, 20000, 30000]
        if save_iterations is None:
            save_iterations = [7000, 10000, 20000, 30000]
        if compress_iterations is None:
            compress_iterations = [7000, 10000, 20000, 30000]
        
        # 使用已有配置文件，通过命令行参数覆盖设置
        # 这样可以避免 Hydra 配置搜索路径的问题
        cmd = [
            sys.executable,
            self.train_script,
            f"--config-name={config_name}",
            f"dataset.source_path={dataset_path}",
            f"dataset.model_path={output_path}",
            f"optimization.iterations={iterations}",
            f"run.use_sh={str(use_sh).lower()}",
            f"run.no_progress_bar=false",
        ]
        
        # 添加测试迭代点
        test_iter_str = "[" + ",".join(map(str, test_iterations)) + "]"
        cmd.append(f"run.test_iterations={test_iter_str}")
        
        # 添加保存迭代点
        save_iter_str = "[" + ",".join(map(str, save_iterations)) + "]"
        cmd.append(f"run.save_iterations={save_iter_str}")
        
        # 添加压缩迭代点
        compress_iter_str = "[" + ",".join(map(str, compress_iterations)) + "]"
        cmd.append(f"run.compress_iterations={compress_iter_str}")
        
        # 添加额外参数
        if additional_args:
            for key, value in additional_args.items():
                cmd.append(f"{key}={value}")
        
        # 切换到SOG目录执行
        cwd = self.sog_path
        
        # 设置环境变量，确保配置文件能被找到
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{self.sog_path}:{env.get('PYTHONPATH', '')}"
        
        # 禁用wandb或设置为离线模式（避免需要API密钥）
        env['WANDB_MODE'] = 'disabled'  # 完全禁用wandb
        # 或者使用离线模式: env['WANDB_MODE'] = 'offline'
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=cwd,
            env=env
        )
        
        return process
    
    def get_available_configs(self) -> list:
        """
        获取可用的配置文件列表
        
        Returns:
            配置文件名列表
        """
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.yaml'):
                configs.append(file[:-5])  # 去掉.yaml后缀
        return sorted(configs)
    
    def get_config_info(self, config_name: str) -> Optional[Dict]:
        """
        获取配置文件信息
        
        Args:
            config_name: 配置文件名（不包括.yaml后缀）
        
        Returns:
            配置字典
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Self-Organizing-Gaussians训练包装器")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="数据集路径（COLMAP格式）"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="输出路径"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="训练迭代次数（默认：30000）"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="ours_q_sh_local_test",
        help="配置文件名（默认：ours_q_sh_local_test）"
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        default=None,
        help="自定义配置文件路径"
    )
    parser.add_argument(
        "--no-sh",
        action="store_true",
        help="禁用球谐函数"
    )
    parser.add_argument(
        "--sog-path",
        type=str,
        default=None,
        help="Self-Organizing-Gaussians路径"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="列出可用的配置文件"
    )
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = SOGTrainer(sog_path=args.sog_path)
    
    # 列出配置文件
    if args.list_configs:
        configs = trainer.get_available_configs()
        print("可用的配置文件：")
        for config in configs:
            print(f"  - {config}")
        return
    
    # 启动训练
    print(f"启动SOG训练...")
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出路径: {args.output_path}")
    print(f"迭代次数: {args.iterations}")
    print(f"配置文件: {args.config_name if args.custom_config is None else args.custom_config}")
    
    process = trainer.train(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        iterations=args.iterations,
        config_name=args.config_name,
        custom_config=args.custom_config,
        use_sh=not args.no_sh
    )
    
    # 实时输出训练日志
    for line in iter(process.stdout.readline, ''):
        if line.strip():
            print(line.strip())
        if "Training complete" in line:
            break
    
    # 等待进程结束
    return_code = process.wait()
    
    if return_code == 0:
        print("\nSOG训练完成！")
    else:
        print(f"\nSOG训练失败，返回码: {return_code}")
        sys.exit(return_code)


if __name__ == "__main__":
    main()

