"""Spawn-isolated RMBench episodes with ordered expert seed selection.

Extract the two phases from the installed upstream loop. Unknown layouts fail
closed, rather than silently changing the benchmark's episode semantics.
"""
import ast
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback


def phase_source(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_policy')
    loop = next(n for n in function.body if isinstance(n, ast.While))
    split = [i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
             and ast.unparse(n.test) == 'TASK_ENV.eval_video_path is not None']
    if len(split) != 2:
        raise RuntimeError('Unsupported RMBench rollout layout')
    index = split[0]
    prefix = copy.deepcopy(function)
    prefix.name = 'select_episodes'
    select_loop = next(n for n in prefix.body if isinstance(n, ast.While))
    select_loop.body = select_loop.body[:index] + ast.parse('''
yield dict(episode_id=now_id, seed=now_seed, instruction=str(instruction),
           episode_info=episode_info, numpy_state=np.random.get_state(),
           python_state=__import__('random').getstate())
TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
if TASK_ENV.render_freq:
    TASK_ENV.viewer.close()
now_id += 1
now_seed += 1
''').body
    rollout = ast.parse('''
def rollout_episode(TASK_ENV, args, model, job, video_size):
    now_id = job['episode_id']
    now_seed = job['seed']
    succ_seed = now_id + 1
    task_name = args['task_name']
    clear_cache_freq = args['clear_cache_freq']
    task_total_reward = 0
    TASK_ENV.suc = 0
    TASK_ENV.test_num = now_id
    args['eval_mode'] = True
    eval_func = eval_function_decorator(args['policy_name'], 'eval')
    reset_func = eval_function_decorator(args['policy_name'], 'reset_model')
    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    TASK_ENV.set_instruction(instruction=job['instruction'])
    np.random.set_state(job['numpy_state'])
    __import__('random').setstate(job['python_state'])
''').body[0]
    rollout.body += copy.deepcopy(loop.body[index:])
    rollout.body += ast.parse("return dict(episode_id=job['episode_id'], seed=now_seed - 1, instruction=job['instruction'], success=succ, reward=task_total_reward)").body
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[prefix, rollout], type_ignores=[])))


def load_upstream(script):
    from run_eval_policy import compile_eval
    ns = {'__file__': script, '__name__': 'rmbench_upstream'}
    source = Path(script).read_text()
    exec(compile_eval(source, script), ns)
    # Extract from the guarded source, retaining bounded grasp retries.
    guarded = ast.unparse(ast.parse(source))
    phases = phase_source(guarded)
    exec(compile_eval(phases, script), ns)
    return ns


def child(role, script, args, model_args, seed, total, video_size, instruction_type, jobs, results, root):
    # Own session also contains planner / ffmpeg descendants for failure cleanup.
    os.setsid()
    name = mp.current_process().name
    work = Path(root) / name
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / 'process.log', 'w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        # Read-only resources remain shared; every relative output is local.
        upstream_root = Path(script).parent.parent
        for entry in upstream_root.iterdir():
            if entry.name in ('assets', 'envs', 'policy', 'description', 'task_config', 'script'):
                (work / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        os.chdir(work)
        sys.path.extend([str(upstream_root), str(upstream_root / 'policy'),
                         str(upstream_root / 'description/utils'), str(upstream_root / 'script')])
        ns = load_upstream(script)
        args = copy.deepcopy(args)
        args['log_file'] = str(work / 'eval_log.txt')
        args['save_path'] = str(work / 'data')
        if args.get('eval_video_log'):
            args['eval_video_save_dir'] = work
        env = ns['class_decorator'](args['task_name'])
        if role == 'selector':
            # Never record expert videos.
            args['eval_video_log'] = False
            args.pop('eval_video_save_dir', None)
            manifest = open(work / 'episodes.jsonl', 'w', buffering=1)
            for job in ns['select_episodes'](args['task_name'], env, args, None, seed,
                                            total, video_size, instruction_type):
                manifest.write(json.dumps({k: job[k] for k in ('episode_id', 'seed', 'instruction', 'episode_info')}, default=str) + '\n')
                jobs.put(job)
            for _ in range(min(int(os.environ['BATCH_SIZE']), total)):
                jobs.put(None)
        else:
            model = ns['eval_function_decorator'](args['policy_name'], 'get_model')(model_args)
            while True:
                job = jobs.get()
                if job is None:
                    break
                result = ns['rollout_episode'](env, args, model, job, video_size)
                results.put(('result', result))
    except BaseException:
        traceback.print_exc()
        results.put(('error', name + ': ' + traceback.format_exc()))
        raise


def parallel_eval(script, task_name, env, args, model_args, st_seed, test_num=100,
                  video_size=None, instruction_type=None):
    ctx = mp.get_context('spawn')
    count = min(int(os.environ['BATCH_SIZE']), test_num)
    root = Path(args['log_file']).parent.resolve()
    jobs, results = ctx.Queue(maxsize=count * 2), ctx.Queue(maxsize=count * 2)
    processes = [ctx.Process(target=child, name=name, args=(role, script, args, model_args,
                 st_seed, test_num, video_size, instruction_type, jobs, results, str(root)))
                 for name, role in [('selector', 'selector')] + [(f'worker_{i}', 'worker') for i in range(count)]]
    completed = {}
    started = time.monotonic()
    def terminate(signum, frame):
        raise RuntimeError(f'Evaluation interrupted by signal {signum}')
    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for process in processes:
            process.start()
        while len(completed) < test_num:
            try:
                kind, value = results.get(timeout=1)
            except queue.Empty:
                if all(p.exitcode is not None for p in processes):
                    raise RuntimeError('All evaluation processes exited before completing episodes')
            else:
                if kind == 'error':
                    raise RuntimeError(value)
                episode_id = value['episode_id']
                if episode_id in completed or not 0 <= episode_id < test_num:
                    raise RuntimeError(f'Duplicate or invalid episode id: {episode_id}')
                completed[episode_id] = value
                (root / f'episode_{episode_id:06d}.json').write_text(json.dumps(value, default=float))
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f'{process.name} crashed: exit={process.exitcode}; see {root / process.name}')
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f'{process.name} failed to finish: {process.exitcode}')
        ordered = [completed[i] for i in range(test_num)]
        if len({r['seed'] for r in ordered}) != test_num:
            raise RuntimeError('Duplicate episode seeds')
        (root / 'episodes.jsonl').write_text(''.join(json.dumps(r, default=float) + '\n' for r in ordered))
        print(f'episodes={test_num} workers={count} episodes_per_second={test_num / (time.monotonic() - started):.4f}', flush=True)
        return max(r['seed'] for r in ordered) + 1, sum(r['success'] for r in ordered), sum(r['reward'] for r in ordered)
    except BaseException:
        (root / 'manager_error.log').write_text(traceback.format_exc())
        raise
    finally:
        for process in processes:
            if process.pid:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    if process.is_alive():
                        process.terminate()
        for process in processes:
            if process.pid:
                process.join(timeout=5)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        jobs.close()
        results.close()


def run(script, code):
    ns = {'__file__': str(script), '__name__': 'rmbench_upstream'}
    exec(code, ns)
    ns['eval_policy'] = lambda *a, **kw: parallel_eval(str(script), *a, **kw)
    # main resolves config; simulation and model clients are constructed only in children.
    ns['class_decorator'] = lambda task: None
    original = ns['eval_function_decorator']
    ns['eval_function_decorator'] = lambda policy, name: (lambda args: args) if name == 'get_model' else original(policy, name)
    ns['main'](ns['parse_args_and_config']())
