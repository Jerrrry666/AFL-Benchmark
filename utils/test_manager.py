"""
异步测试管理器
允许测试在后台独立执行，不影响训练进度
"""

import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Callable
import torch
import numpy as np


class TestManager:
    """
    异步测试管理器
    - 管理测试任务队列
    - 在独立线程中执行测试
    - 支持测试结果回调
    - 避免与训练争抢GPU资源
    """
    
    def __init__(self, 
                 server,
                 max_workers: int = 2,
                 test_interval: float = 0.5,
                 enable_callback: bool = True):
        """
        Args:
            server: FedSim服务器实例
            max_workers: 最大测试工作线程数
            test_interval: 测试轮询间隔（秒）
            enable_callback: 是否启用结果回调
        """
        self.server = server
        self.max_workers = max_workers
        self.test_interval = test_interval
        self.enable_callback = enable_callback
        
        # 测试队列和状态
        self.test_queue = queue.Queue()
        self.test_results = {}
        self.pending_tests = set()
        self.test_history = []
        
        # 线程控制
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TestWorker")
        self.test_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger("TestManager")
        
        # 结果回调
        self.callbacks: List[Callable] = []
        
        # 资源控制
        self.gpu_semaphore = threading.Semaphore(max_workers)  # 控制GPU使用
        self.current_test_round = 0
        
    def start(self):
        """启动测试管理器"""
        if self.test_thread is None or not self.test_thread.is_alive():
            self.stop_event.clear()
            self.test_thread = threading.Thread(target=self._test_worker, daemon=True)
            self.test_thread.start()
            self.logger.info("测试管理器已启动")
    
    def stop(self):
        """停止测试管理器"""
        if self.test_thread and self.test_thread.is_alive():
            self.stop_event.set()
            self.test_thread.join(timeout=5.0)
            self.logger.info("测试管理器已停止")
    
    def submit_test(self, round_id: int, priority: int = 0, force: bool = False) -> str:
        """
        提交测试任务
        
        Args:
            round_id: 训练轮次ID
            priority: 优先级（数值越大优先级越高）
            force: 是否强制执行（跳过频率限制）
        
        Returns:
            str: 任务ID
        """
        task_id = f"test_round_{round_id}_{time.time()}"
        
        test_task = {
            'task_id': task_id,
            'round_id': round_id,
            'priority': priority,
            'force': force,
            'timestamp': time.time()
        }
        
        self.test_queue.put((priority, test_task))
        self.pending_tests.add(task_id)
        
        self.logger.debug(f"已提交测试任务: {task_id}")
        return task_id
    
    def should_test(self, round_id: int, test_gap: int, force: bool = False) -> bool:
        """
        判断是否应该执行测试
        
        Args:
            round_id: 当前训练轮次
            test_gap: 测试间隔
            force: 是否强制测试
        """
        if force:
            return True
            
        # 检查是否为测试轮次
        if (self.server.total_round - round_id <= 10) or (round_id % test_gap == (test_gap - 1)):
            return True
            
        return False
    
    def get_test_result(self, task_id: str, timeout: float = 1.0) -> Optional[Dict]:
        """
        获取测试结果（异步）
        
        Args:
            task_id: 任务ID
            timeout: 等待超时时间
        
        Returns:
            Dict: 测试结果，None如果任务未完成
        """
        if task_id in self.test_results:
            return self.test_results[task_id]
        return None
    
    def add_callback(self, callback: Callable):
        """添加测试结果回调函数"""
        self.callbacks.append(callback)
    
    def _test_worker(self):
        """测试工作线程"""
        while not self.stop_event.is_set():
            try:
                # 等待测试任务
                try:
                    priority, task = self.test_queue.get(timeout=self.test_interval)
                except queue.Empty:
                    continue
                
                task_id = task['task_id']
                round_id = task['round_id']
                
                self.logger.debug(f"开始执行测试任务: {task_id}")
                
                # 执行测试
                test_result = self._execute_test(round_id, task_id)
                
                if test_result:
                    # 存储结果
                    self.test_results[task_id] = test_result
                    self.test_history.append({
                        'round_id': round_id,
                        'result': test_result,
                        'timestamp': time.time()
                    })
                    
                    # 触发回调
                    if self.enable_callback and self.callbacks:
                        for callback in self.callbacks:
                            try:
                                callback(round_id, test_result)
                            except Exception as e:
                                self.logger.error(f"回调函数执行失败: {e}")
                    
                    self.logger.info(f"[Round {round_id}] 异步测试完成 - Acc: {test_result.get('acc', 0):.2f}")
                
                # 从待处理集合中移除
                self.pending_tests.discard(task_id)
                
            except Exception as e:
                self.logger.error(f"测试工作线程错误: {e}")
                time.sleep(1.0)  # 避免快速重试
    
    def _execute_test(self, round_id: int, task_id: str) -> Optional[Dict]:
        """
        执行实际的测试任务
        
        Args:
            round_id: 训练轮次
            task_id: 任务ID
        
        Returns:
            Dict: 测试结果
        """
        try:
            # 控制GPU资源访问
            with self.gpu_semaphore:
                # 确保测试在合适的设备上执行
                original_device = getattr(self.server, 'device', 'cpu')
                
                # 切换到测试设备
                if hasattr(self.server, 'test_device'):
                    self.server.device = self.server.test_device
                else:
                    self.server.device = 'cpu'  # 强制使用CPU进行测试，避免与训练争抢GPU
                
                # 执行测试
                test_result = self.server.test_all()
                
                # 恢复原始设备
                self.server.device = original_device
                
                # 添加任务信息
                test_result['task_id'] = task_id
                test_result['round_id'] = round_id
                test_result['wall_clock_time'] = self.server.wall_clock_time
                
                return test_result
                
        except Exception as e:
            self.logger.error(f"测试执行失败 (Round {round_id}): {e}")
            return None
    
    def get_status(self) -> Dict:
        """获取测试管理器状态"""
        return {
            'is_running': self.test_thread.is_alive() if self.test_thread else False,
            'queue_size': self.test_queue.qsize(),
            'pending_count': len(self.pending_tests),
            'completed_count': len(self.test_results),
            'history_count': len(self.test_history)
        }
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.stop()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class TestConfig:
    """测试配置管理"""
    
    def __init__(self):
        self.default_test_gap = 5
        self.max_concurrent_tests = 2
        self.test_timeout = 30.0
        self.enable_async = True
        
    def should_run_test(self, round_id: int, total_rounds: int, test_gap: int) -> bool:
        """判断是否应该运行测试"""
        return (total_rounds - round_id <= 10) or (round_id % test_gap == (test_gap - 1))
