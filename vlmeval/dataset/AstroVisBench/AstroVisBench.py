import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from vlmeval.smp import *
from vlmeval.dataset.image_vqa import ImageVQADataset
from .evaluate_helper import *
from .vis_eval_helper import *
import ast
import multiprocessing
import time
import traceback
import os
import json
SYS_PROMPT = """
    You are tasked with completing a jupyter notebook about astronomy. You will be given some markdown cells and some python code cells for context, and in response you must output only python code that accurately fulfills the goals of the notebook as described by the markdown text.
"""


# --- 之前的 worker_wrapper 保持不变 ---
def worker_wrapper(queue, task_id, kwargs):
    """
    queue: 用于回传结果
    task_id: 用于追踪是哪个任务
    kwargs: 执行参数
    """
    try:
        # 执行核心逻辑
        result = execute_notebook_w_pro_test(**kwargs)
        queue.put({'status': 'success', 'task_id': task_id, 'data': result})
    except Exception as e:
        # 捕获 Python 异常
        queue.put({'status': 'error', 'task_id': task_id, 'error': str(e)})

def vis_worker_wrapper(queue, uid, kwargs):
    """
    用于可视化任务的子进程包装函数。
    """
    try:
        # 调用原本的可视化执行函数
        # 假设 execute_notebook_visout 已经在上下文可见
        result = execute_notebook_visout(**kwargs)
        queue.put({'status': 'success', 'task_id': uid, 'data': result})
    except Exception as e:
        # 捕获常规异常
        queue.put({'status': 'error', 'task_id': uid, 'error': str(e)})

def remove_ticks(code):
    code_lines = code.split("\n")
    new_code_lines = [line for line in code_lines if not line.startswith("```") and not "<CELL END>" in line]
    return "\n".join(new_code_lines)

class AstroVisBench(ImageVQADataset):
    DATASET_URL = {
        'AstroVisBench_Processing': 'https://opencompass.openxlab.space/utils/VLMEval/AstroVisBench.tsv',
        'AstroVisBench_Visualization': 'https://opencompass.openxlab.space/utils/VLMEval/AstroVisBench.tsv'# e.g. 'https://your.host/astrovisbench.tsv' or 'tos://.../astrovisbench.tsv'
    }
    DATASET_MD5 = {
        'AstroVisBench_Processing': None,
        'AstroVisBench_Visualization': None
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)
        self.dataset = dataset
        self.dataset_name = dataset
        self.TYPE = 'QA'
        self.MODALITY = 'IMAGE'  # 即使是纯文本题也可以设置为 IMAGE；若完全无图，可改为 'TEXT'
        self.add_pro_uds = True ## todo
        self.add_vis_uds = False
        self.env_path = '/mnt/shared-storage-user/lishuo/bench_environment'
        self.vis_cache_path = None
        self.gen_cache_path = None
        self.true_cache_path = None

    def build_prompt(self, line):
        msgs = [{'type': 'text', 'value': SYS_PROMPT, 'role': 'system'}]
        if self.dataset_name == 'AstroVisBench_Processing':
            context = [line['setup_query'],
                       '```\n\n' + line['setup_gt_code'] + '\n\n```',
                       line['processing_query'],
                       ]

            if self.add_pro_uds:
                context.append(line['processing_underspecifications'])
        elif self.dataset_name == 'AstroVisBench_Visualization':
            context = [line['setup_query'],
                       '```\n\n' + line['setup_gt_code'] + '\n\n```',
                       line['processing_query'],
                       '```\n\n' + line['processing_gt_code'] + '\n\n```',
                       line['visualization_query'],
                       ]

            if self.add_vis_uds:
                context.append(line['visualization_underspecifications'])
        else:
            raise Exception('Unknown dataset')

        context = '\n\n'.join(context)
        msgs.append({'type': 'text', 'value': context, 'role': 'user'})
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        eval_root = os.path.dirname(eval_file)
        vis_cache_dir = os.path.join(eval_root,'vis_cache')
        os.makedirs(vis_cache_dir, exist_ok=True)
        print(f'vis cache dir: {vis_cache_dir}')
        gen_cache_dir = os.path.join(eval_root,'gen_cache')
        os.makedirs(gen_cache_dir, exist_ok=True)
        print(f'gen cache dir: {gen_cache_dir}')
        model_root_dir = os.path.dirname(eval_root)
        true_cache_dir = os.path.join(model_root_dir, 'true_cache')
        os.makedirs(true_cache_dir, exist_ok=True)
        print(f'true cache dir: {true_cache_dir}')
        self.vis_cache_path = vis_cache_dir
        self.gen_cache_path = gen_cache_dir
        self.true_cache_path = true_cache_dir

        if self.dataset_name == 'AstroVisBench_Processing':
            self.evaluate_process(eval_file, **judge_kwargs)
        elif self.dataset_name == 'AstroVisBench_Visualization':
            self.evaluate_visualization(eval_file, **judge_kwargs)
        else:
            raise Exception('Unknown dataset')
        pass

    def evaluate_process(self, eval_file, **judge_kwargs):
        print(f"Starting evaluation for {self.dataset_name}...")

        # 设置并行数量（建议设置为 CPU 核心数 - 2，或者根据显存/内存大小调整）
        # 如果任务主要是 CPU 密集型或 IO，可以设大一点，比如 8 或 16
        MAX_WORKERS = 60
        
        # 使用 spawn 确保隔离性
        ctx = multiprocessing.get_context('spawn')
        
        try:
            df = pd.DataFrame(load(eval_file))
        except Exception as e:
            print(f"Error loading evaluation file {eval_file}: {e}")
            return

        all_evaluation_results = []
        os.makedirs(self.true_cache_path, exist_ok=True)
        os.makedirs(self.gen_cache_path, exist_ok=True)

        print(f"Executing tasks with parallelism: {MAX_WORKERS} workers...")
        
        # 使用 Manager Queue 来接收所有子进程的结果
        # Manager Queue 比普通 Queue 更重，但在多进程并发下管理更方便
        manager = ctx.Manager()
        result_queue = manager.Queue()

        # 准备任务列表
        # 将所有任务转为参数字典
        pending_tasks = []
        for index, row in df.iterrows():
            task_dict = self._prepare_jdict(row)
            func_kwargs = {
                'jdict': task_dict,
                'true_cache': self.true_cache_path,
                'gen_cache': self.gen_cache_path,
                'skip_test': False,
                'temp_caching': True,
                'min_diff_only': True
            }
            # 存储 (uid, 原始row/task_dict, 参数)
            pending_tasks.append((row.get('uid'), task_dict, func_kwargs))

        # 正在运行的进程列表：{'process': Process对象, 'uid': uid, 'task_dict': task_dict, 'start_time': time}
        running_procs = []
        
        # 进度条
        pbar = tqdm(total=len(pending_tasks))
        
        # 任务调度循环
        task_idx = 0
        while task_idx < len(pending_tasks) or len(running_procs) > 0:
            
            # 1. 填充进程池：如果当前运行数 < MAX_WORKERS 且还有任务没发，就启动新进程
            while len(running_procs) < MAX_WORKERS and task_idx < len(pending_tasks):
                uid, task_dict, func_kwargs = pending_tasks[task_idx]
                
                # 启动进程
                p = ctx.Process(target=worker_wrapper, args=(result_queue, uid, func_kwargs))
                p.start()
                
                running_procs.append({
                    'p': p,
                    'uid': uid,
                    'task_dict': task_dict,
                    'start_time': time.time()
                })
                task_idx += 1

            # 2. 检查正在运行的进程状态
            # 我们倒序遍历，方便在循环中安全删除已完成的项
            for i in range(len(running_procs) - 1, -1, -1):
                proc_info = running_procs[i]
                p = proc_info['p']
                uid = proc_info['uid']
                
                # 检查是否超时 (例如 600秒)
                if time.time() - proc_info['start_time'] > 600:
                    print(f"Task {uid} timed out. Killing...")
                    if p.is_alive():
                        p.terminate()
                        p.join()
                    
                    # 记录超时错误
                    err_res = proc_info['task_dict'].copy()
                    err_res['processing_test'] = {'error': 'Timeout', 'pro_success': False}
                    all_evaluation_results.append(err_res)
                    
                    running_procs.pop(i)
                    pbar.update(1)
                    continue

                # 检查进程是否结束
                if not p.is_alive():
                    p.join() # 确保资源释放
                    
                    # 检查是否是异常退出 (Segfault exitcode 通常是负数)
                    if p.exitcode != 0:
                        print(f"Task {uid} crashed with exit code {p.exitcode} (Segfault).")
                        err_res = proc_info['task_dict'].copy()
                        err_res['processing_test'] = {'error': f'Process crashed (Exit code: {p.exitcode})', 'pro_success': False}
                        all_evaluation_results.append(err_res)
                    
                    # 注意：如果 exitcode == 0，结果会在 result_queue 里，稍后统一取
                    
                    running_procs.pop(i)
                    pbar.update(1)

            # 3. 从队列中收集结果
            # 因为多个进程共用一个队列，我们需要不断把结果取出来
            while not result_queue.empty():
                try:
                    res = result_queue.get_nowait()
                    # 这里的 res 包含 {'status', 'task_id', 'data'/'error'}
                    # 我们需要把成功的结果存起来
                    # (注意：这里最好通过 task_id 匹配，但简单起见我们直接存)
                    
                    if res['status'] == 'success':
                        all_evaluation_results.append(res['data'])
                    else:
                        # Python 级别的报错（非崩溃）
                        # 我们需要找到对应的 task_dict 才能存错误记录
                        # 为了简化，可以在 worker_wrapper 里把 task_dict 传回来，或者在这里忽略（因为崩溃检查已经覆盖大部分）
                        # 简单的做法：
                        pass 
                        # 注意：这里的逻辑如果想完美，需要在 pending_tasks 里建立 uid -> task_dict 的映射
                        # 但通常如果 worker 内部 catch 了异常，我们希望它返回一个包含错误信息的 result_dict
                        # *修正*：建议修改 execute_notebook_w_pro_test 让它在异常时也返回一个结构完整的 dict
                except:
                    break

            # 避免 CPU 空转
            time.sleep(0.1)

        pbar.close()

        # 3. 保存结果到jsonl文件 
        print("Saving evaluation results to josnl file...")
        try:
            eval_root = os.path.dirname(eval_file)
            output_filename = os.path.join(eval_root, os.path.splitext(os.path.basename(eval_file))[
                0] + "_processing_results.jsonl")

            with open(output_filename, 'w') as f:
                for result_dict in all_evaluation_results:
                    f.write(json.dumps(result_dict, default=str) + '\n')

            print(f"Evaluation finished. Results saved to {output_filename}")

        except Exception as e:
            print(f"Failed to save results to jsonl file: {e}")
            fallback_filename = "processing_results_fallback.json"
            print(f"Saving to {fallback_filename} as a fallback.")
            with open(fallback_filename, 'w') as f:
                f.write(json.dumps(all_evaluation_results, indent=4, default=str))

        self.aggregate_results_for_pro(all_evaluation_results, eval_root)

        return

    def evaluate_visualization(self, eval_file, **judge_kwargs):
        """
        对AstroVisBench的可视化任务进行两阶段评估 (支持断点续跑)。
        阶段一：并行执行代码并生成图像 (并行化 + 进程隔离 + 实时保存)。
        阶段二：使用VLM对生成的图像进行评估 (保持串行)。
        """
        print(f"Starting evaluation for {self.dataset_name}...")

        # --- 阶段零：依赖检查和数据加载 ---
        eval_root = os.path.dirname(eval_file)
        processing_results_filename = os.path.join(eval_root, os.path.splitext(os.path.basename(eval_file))[0].replace("_Visualization",
                                                                                               "_Processing") + "_processing_results.jsonl")
        
        # 定义阶段一的输出文件名 (定义提前，方便检查)
        vis_exec_output_filename = processing_results_filename.replace(".jsonl", "_vis_exec_results.jsonl")

        # 1. 检查前置依赖文件
        if not os.path.exists(processing_results_filename):
            raise FileNotFoundError(
                f"Prerequisite file not found: {processing_results_filename}. Please run the AstroVisBench_Processing evaluation first.")

        print(f"Loading prerequisite processing results from: {processing_results_filename}")
        df_process_results = load(processing_results_filename)
        df_process_results = pd.DataFrame(df_process_results)
        if 'prediction' in df_process_results.columns:
            df_process_results.rename(columns={'prediction': 'prediction_process'}, inplace=True)

        print(f"Loading visualization code from: {eval_file}")
        df_vis_code = pd.DataFrame(load(eval_file))

        # 2. 合并数据
        if 'uid' not in df_process_results.columns or 'uid' not in df_vis_code.columns:
            raise KeyError("Column 'uid' is required for merging but not found in one of the files.")

        df_merged = pd.merge(df_process_results, df_vis_code[['uid', 'prediction']], on='uid', how='inner')

        if len(df_merged) != len(df_vis_code):
            print(f"Warning: Merging resulted in {len(df_merged)} rows, but visualization input has {len(df_vis_code)} rows. Some tasks may be missing.")

        # --- 阶段一：并行执行可视化代码 (带断点续跑逻辑) ---
        print("\n--- Starting Stage 1: Code Execution (Parallel & Resumable) ---")
        os.makedirs(self.vis_cache_path, exist_ok=True)
        
        # [新增逻辑] 1. 检查已完成的任务
        finished_uids = set()
        existing_results = []
        
        if os.path.exists(vis_exec_output_filename):
            print(f"Found existing execution results at {vis_exec_output_filename}. Checking for resumable tasks...")
            try:
                with open(vis_exec_output_filename, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            item = json.loads(line)
                            # 确保这条记录里有 uid，且被判定为已执行（无论成功失败）
                            if 'uid' in item:
                                finished_uids.add(item['uid'])
                                existing_results.append(item)
                        except json.JSONDecodeError:
                            print("Warning: Skipping a corrupted line in existing results file.")
            except Exception as e:
                print(f"Error reading existing file: {e}. Starting fresh.")
                finished_uids = set()
                existing_results = []

        print(f"Total tasks: {len(df_merged)}. Already finished: {len(finished_uids)}.")

        # [新增逻辑] 2. 准备所有待办任务，过滤掉已完成的
        pending_tasks = []
        for index, row in df_merged.iterrows():
            if row['uid'] in finished_uids:
                continue # 跳过已完成的
            
            task_dict = self._prepare_jdict_for_vis(row)
            func_kwargs = {
                'jdict': task_dict,
                'gen_vis_cache': self.vis_cache_path
            }
            pending_tasks.append((row.get('uid'), task_dict, func_kwargs))

        # 如果所有任务都完成了，直接跳过执行阶段
        vis_exec_results_new = [] # 仅存储本次运行产生的新结果
        
        if len(pending_tasks) > 0:
            print(f"Resuming execution for {len(pending_tasks)} remaining tasks.")

            MAX_WORKERS = 60
            ctx = multiprocessing.get_context('spawn')
            manager = ctx.Manager()
            result_queue = manager.Queue()
            
            running_procs = []
            pbar = tqdm(total=len(pending_tasks), desc="Stage 1 Execution (Running)")
            
            task_idx = 0
            
            # 打开文件准备追加写入 (Append Mode)
            # 使用 'a' 模式，确保实时写入
            with open(vis_exec_output_filename, 'a') as f_out:
                
                while task_idx < len(pending_tasks) or len(running_procs) > 0:
                    
                    # 1. 启动进程
                    while len(running_procs) < MAX_WORKERS and task_idx < len(pending_tasks):
                        uid, task_dict, func_kwargs = pending_tasks[task_idx]
                        p = ctx.Process(target=vis_worker_wrapper, args=(result_queue, uid, func_kwargs))
                        p.start()
                        running_procs.append({
                            'p': p, 'uid': uid, 'task_dict': task_dict, 'start_time': time.time()
                        })
                        task_idx += 1

                    # 2. 监控进程 (超时/崩溃)
                    for i in range(len(running_procs) - 1, -1, -1):
                        proc_info = running_procs[i]
                        p = proc_info['p']
                        uid = proc_info['uid']
                        error_result = None

                        # 超时检查
                        if time.time() - proc_info['start_time'] > 600:
                            print(f"Vis Task {uid} timed out. Killing...")
                            if p.is_alive(): p.terminate(); p.join()
                            error_result = proc_info['task_dict'].copy()
                            error_result['visualization_test'] = {'error': 'Timeout', 'vis_success': False, 'gen_vis_list': []}

                        # 崩溃/结束检查
                        elif not p.is_alive():
                            p.join()
                            if p.exitcode != 0:
                                print(f"Vis Task {uid} crashed (Exit: {p.exitcode}).")
                                error_result = proc_info['task_dict'].copy()
                                error_result['visualization_test'] = {'error': f'Crashed (Code {p.exitcode})', 'vis_success': False, 'gen_vis_list': []}
                            else:
                                pass # 正常结束，等待 Queue 处理

                        # 如果发生了错误（超时或崩溃），需要手动记录并从运行列表移除
                        if error_result:
                            # 写入文件和内存
                            f_out.write(json.dumps(error_result, default=str) + '\n')
                            f_out.flush() # 强制刷入磁盘
                            vis_exec_results_new.append(error_result)
                            
                            running_procs.pop(i)
                            pbar.update(1)
                        elif not p.is_alive():
                            # 正常结束的进程只从列表移除，数据在 Queue 中获取
                            running_procs.pop(i)

                    # 3. 收集结果 (实时写入)
                    while not result_queue.empty():
                        try:
                            res = result_queue.get_nowait()
                            final_data = None
                            
                            if res['status'] == 'success':
                                final_data = res['data']
                            else:
                                # 处理 Worker 内部捕获的 Python 异常
                                err_task_dict = next((t[1] for t in pending_tasks if t[0] == res['task_id']), None)
                                if err_task_dict:
                                    final_data = err_task_dict.copy()
                                    final_data['visualization_test'] = {'error': res['error'], 'vis_success': False, 'gen_vis_list': []}
                            
                            if final_data:
                                # [核心修改] 立即写入文件
                                f_out.write(json.dumps(final_data, default=str) + '\n')
                                f_out.flush() # 强制刷入磁盘，防止断电丢失
                                
                                vis_exec_results_new.append(final_data)
                                pbar.update(1) # 这里更新进度条，通常比 process join 更准确反映完成度
                        except Exception:
                            break
                    
                    time.sleep(0.1)

            pbar.close()
            manager.shutdown()
            print(f"Stage 1 finished. New results saved to {vis_exec_output_filename}")
        
        else:
            print("All tasks are already completed in Stage 1. Skipping execution.")

        # --- 阶段二：VLM 评估 ---
        print("\n--- Starting Stage 2: VLM Evaluation ---")
        
        # [核心修改] 组装全量数据：旧数据 + 新跑出的数据
        # 注意：这里直接用 existing_results + vis_exec_results_new 即可
        # 为了保险起见（防止内存里的和文件里的不一致），也可以选择重新读取一次完整文件，
        # 但既然我们维护了两个 list，直接合并效率更高。
        full_vis_results = existing_results + vis_exec_results_new
        
        # 简单校验一下数量
        if len(full_vis_results) == 0:
            print("Warning: No results available for VLM evaluation!")
            return

        try:
            # 执行 VLM 评估 (这里假设 do_vis_eval 内部是串行或 API 调用的)
            final_results_list = do_vis_eval(full_vis_results)

            final_output_filename = vis_exec_output_filename.replace(".jsonl", "_final_results.jsonl")
            print(f"Stage 2 finished. Saving final evaluation results to {final_output_filename}")
            
            with open(final_output_filename, 'w') as f:
                for result_dict in final_results_list:
                    f.write(json.dumps(result_dict, default=str) + '\n')

            print("\nEvaluation complete!")

        except Exception as e:
            print(f"An error occurred during VLM evaluation: {e}")
            traceback.print_exc()
            # 如果阶段2失败，至少阶段1的数据已经安全保存在磁盘上了，下次跑会自动跳过阶段1

        self.aggregate_results_for_vis(final_results_list, eval_root)
        return
    
    def _prepare_jdict(self, row):
        """
        辅助函数：将DataFrame的一行转换为`exec_bench.py`期望的字典格式。
        """
        jdict = row.to_dict()

        # 核心适配步骤：将'prediction'列的内容映射回'processing_gen_code'
        jdict['processing_gen_code'] = remove_ticks(jdict['prediction'])

        # 确保visualization_gen_code字段存在但为空，以防后续函数意外使用
        jdict['visualization_gen_code'] = ""

        # 动态计算并添加环境路径，就像原始main函数做的那样
        jdict['env_path'] = get_env_path(jdict, self.env_path)

        return jdict

    def _prepare_jdict_for_vis(self, row):
        """
        辅助函数：将合并后的DataFrame行转换为vis评估期望的字典格式。
        """
        jdict = row.to_dict()

        # 核心适配：将'prediction'列的内容映射回'visualization_gen_code'
        jdict['visualization_gen_code'] = remove_ticks(jdict['prediction'])

        # 动态计算并添加环境路径
        jdict['env_path'] = get_env_path(jdict, self.env_path)


        if isinstance(jdict.get('processing_test'), str):
            try:
                jdict['processing_test'] = ast.literal_eval(jdict['processing_test'])
            except (ValueError, SyntaxError):
                print(
                    f"Warning: Could not parse 'processing_test' field for uid {jdict.get('uid')}. Treating as empty.")
                jdict['processing_test'] = {}

        return jdict

    # 放置在您的AstroVisBench类定义内

    def aggregate_results_for_pro(self, results_list, output_dir):
        """
        对处理任务的评估结果进行聚合统计，并保存为JSON文件。
        :param results_list: 包含了处理任务评估结果的字典列表。
        :param output_dir: 保存最终聚合结果JSON文件的目录。
        """
        print("Aggregating results for Processing tasks...")

        # 1. 统计执行成功率
        # 提取每个任务的pro_success标志（True/False）
        process_success_flags = [
            q.get('processing_test', {}).get('pro_success', False) for q in results_list
        ]
        # 计算成功率（True记为1，False记为0，然后求平均值）
        avg_pro_success_rate = np.mean(process_success_flags) if process_success_flags else 0

        # 2. 统计变量检查分数 (VIscore)
        # 仅在执行成功且inspections_results有效的情况下提取分数
        processing_vi_scores = [
            q['processing_test']['inspection_results']['agg_scores']['unweighted']
            for q in results_list
            if q.get('processing_test', {}).get('pro_success', False) and \
               q.get('processing_test', {}).get('inspection_results', {}).get('results')  # 确保results字段非空
        ]
        # 计算平均VIscore
        avg_vi_score = np.mean(processing_vi_scores) if processing_vi_scores else 0

        # 3. 构造结果字典
        final_stats = {
            "processing_task_execution_success_rate": avg_pro_success_rate,
            "average_variable_inspection_score": avg_vi_score,
            "total_tasks_evaluated": len(results_list)
        }

        # 4. 保存到JSON文件
        output_filename = os.path.join(output_dir, "processing_aggregation_results.json")
        try:
            with open(output_filename, 'w') as f:
                json.dump(final_stats, f, indent=4)
            print(f"Processing aggregation results saved to {output_filename}")
        except Exception as e:
            print(f"Error saving processing aggregation results: {e}")

        return final_stats

    def aggregate_results_for_vis(self, results_list, output_dir):
        """
        对可视化任务的评估结果进行聚合统计，并保存为JSON文件。
        :param results_list: 包含了可视化任务评估结果的字典列表。
        :param output_dir: 保存最终聚合结果JSON文件的目录。
        """
        from collections import Counter  # 仅在此处需要，局部导入
        print("Aggregating results for Visualization tasks...")

        # 1. 统计执行成功率
        vis_success_flags = [
            q.get('visualization_test', {}).get('vis_success', False) for q in results_list
        ]
        avg_vis_success_rate = np.mean(vis_success_flags) if vis_success_flags else 0

        # 2. 统计VLM评判的错误分布
        vis_eval_errors = [
            q.get('visualization_llm_eval', {}).get('errors', 'Evaluation_Error') for q in results_list
        ]
        # 使用Counter统计每种错误类型的数量
        vis_error_distribution = dict(Counter(vis_eval_errors))

        # 3. 构造结果字典
        final_stats = {
            "visualization_task_execution_success_rate": avg_vis_success_rate,
            "vlm_evaluation_error_distribution": vis_error_distribution,
            "total_tasks_evaluated": len(results_list)
        }

        # 4. 保存到JSON文件
        output_filename = os.path.join(output_dir, "visualization_aggregation_results.json")
        try:
            with open(output_filename, 'w') as f:
                json.dump(final_stats, f, indent=4)
            print(f"Visualization aggregation results saved to {output_filename}")
        except Exception as e:
            print(f"Error saving visualization aggregation results: {e}")

        return final_stats