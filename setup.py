#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup.py —— 一键环境准备 / 建库 / 启动（跨平台）

设计原则
========
1. **本文件只用标准库**，所以随便哪个 Python 都能跑起来（包括系统 Python）。
   需要第三方依赖的活儿（下载、转换、建库）一律交给 `.venv` 里的解释器执行。
2. **幂等**：每一步都先检查产物在不在，在就跳过。可以反复跑。
3. **不联网就不硬失败**：下载类步骤失败会给出明确的手动做法，而不是抛栈。

子命令
======
    python setup.py check     检查当前状态（只读，不改任何东西）
    python setup.py venv      建虚拟环境 + 装依赖
    python setup.py models    下载 embedding / 精排模型
    python setup.py data      下载开源数据集并转换合并
    python setup.py ingest    建库（2 万多切块，约 25 分钟）
    python setup.py serve     启动 HTTP 服务并打开浏览器
    python setup.py all       venv + models + data + ingest

Windows 用户直接双击 `启动.bat` 即可，不用记这些命令。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / '.venv'
_BIN = 'Scripts' if os.name == 'nt' else 'bin'
_EXE = 'python.exe' if os.name == 'nt' else 'python'
PY = VENV / _BIN / _EXE
PYPI_MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple'

# pip 源候选，按顺序回退。
#
# ⚠️ 为什么必须回退（2026-10-09 实测）
#    清华镜像会**偶发抽风**：索引页拉取失败时 pip 只报
#        ERROR: Could not find a version that satisfies the requirement ...
#        ERROR: No matching distribution found for ...
#    看起来像"包不存在"或"Python 版本不支持"，**完全看不出是镜像的问题**。
#    实测同一个 venv、同一条命令，几分钟前全部失败、几分钟后全部成功。
#    所以一键脚本必须能换源重试，否则使用者会卡在一个误导性的错误上。
PYPI_INDEXES = [
    ('清华', 'https://pypi.tuna.tsinghua.edu.cn/simple'),
    ('阿里云', 'https://mirrors.aliyun.com/pypi/simple/'),
    ('中科大', 'https://pypi.mirrors.ustc.edu.cn/simple/'),
    ('官方', None),
]

DB = ROOT / 'data' / 'rag.db'
CORPUS = ROOT / 'data' / 'corpus.json'
XLSX = ROOT / 'data' / 'external' / 'job_posting_data.xlsx'


# ─────────────── 基础设施 ───────────────

def _win_user_env(name: str) -> str | None:
    """
    从 Windows 注册表读**用户级**环境变量。

    为什么要这个：用户的 DEEPSEEK_API_KEY 通常是"用户变量"，
    只有新开的进程才继承。如果当前 shell 是在设置之前启动的，
    os.environ 里就没有它——表现出来就是 /ask 说"未设置 API Key"，
    但用户明明配过。这里兜一下，只读、不写。
    """
    if os.name != 'nt':
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
            v, _ = winreg.QueryValueEx(k, name)
            return str(v) if v else None
    except Exception:                                              # noqa: BLE001
        return None


def child_env() -> dict:
    """子进程环境：HF 镜像 + 关闭 Xet（否则走 cas-server.xethub.hf.co 会 401）。"""
    e = dict(os.environ)
    e.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    e.setdefault('HF_HUB_DISABLE_XET', '1')
    e.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    e['PYTHONIOENCODING'] = 'utf-8'
    if not e.get('DEEPSEEK_API_KEY'):
        v = _win_user_env('DEEPSEEK_API_KEY')
        if v:
            e['DEEPSEEK_API_KEY'] = v
    return e


def run(args, *, quiet=False, check=True) -> int:
    args = [str(a) for a in args]
    if not quiet:
        print('  $ %s' % ' '.join(args))
    rc = subprocess.call(args, env=child_env(), cwd=str(ROOT))
    if check and rc != 0:
        print('  [!] 上一条命令返回 %d' % rc)
    return rc


def hr(title: str) -> None:
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


def need_venv() -> bool:
    if PY.is_file():
        return False
    print('  [!] 还没建虚拟环境。先跑：python setup.py venv')
    return True


# ─────────────── 各步骤 ───────────────

def pip_install_requirements() -> int:
    """
    装依赖，**逐源回退**。

    额外加了 --retries/--timeout 让坏源尽快暴露，而不是卡十分钟。
    """
    base = [PY, '-m', 'pip', 'install', '-r', 'requirements.txt',
            '--no-cache-dir', '--retries', '2', '--timeout', '30']
    last = 1
    for name, url in PYPI_INDEXES:
        args = list(base) + (['-i', url] if url else [])
        print('  用 %s 源安装 …' % name)
        last = run(args, quiet=True, check=False)
        if last == 0:
            return 0
        print('  [!] %s 源失败，换下一个' % name)
    return last


def step_venv() -> int:
    hr('[1/5] 虚拟环境与依赖')
    if PY.is_file():
        print('  .venv 已存在，跳过创建：%s' % PY)
    else:
        print('  创建 .venv（用 %s）' % sys.executable)
        if sys.version_info < (3, 10):
            print('  [错误] 需要 Python 3.10 或更高，当前 %s' % sys.version.split()[0])
            return 1
        rc = run([sys.executable, '-m', 'venv', str(VENV)])
        if rc != 0 or not PY.is_file():
            print('  [错误] 创建虚拟环境失败。确认 `python -m venv` 可用。')
            return 1
    print('  安装依赖（默认源在国内会卡十几分钟，所以优先走镜像）…')
    if pip_install_requirements() != 0:
        print()
        print('  [错误] 所有 pip 源都失败了。可以手动试：')
        print('    %s -m pip install -r requirements.txt' % PY)
        print('  或者指定自己能用的源：')
        print('    %s -m pip install -r requirements.txt -i <你的源>/simple' % PY)
        return 1
    print('  [OK] 依赖就绪')
    return 0


def step_models() -> int:
    hr('[2/5] 下载模型')
    if need_venv():
        return 1
    script = (
        'import os\n'
        'os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")\n'
        'os.environ.setdefault("HF_HUB_DISABLE_XET","1")\n'
        'from fastembed import TextEmbedding\n'
        'from fastembed.rerank.cross_encoder import TextCrossEncoder\n'
        'print("  embedding …")\n'
        'TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")\n'
        'print("  reranker  …（约 0.6 GB，第一次会慢）")\n'
        'TextCrossEncoder(model_name="BAAI/bge-reranker-v2-m3-int8")\n'
        'print("  模型就绪")\n'
    )
    return run([PY, '-u', '-c', script])


def step_data() -> int:
    hr('[3/5] 语料')
    if need_venv():
        return 1
    if CORPUS.is_file():
        size = CORPUS.stat().st_size / 1048576
        print('  data/corpus.json 已存在（%.2f MB），跳过。' % size)
        print('  想重建就删掉它再跑一次。')
        return 0
    print('  下载开源数据集（github.com 直连不通，脚本会走镜像）…')
    if run([PY, '-u', '-B', 'src/fetch_external_jds.py']) != 0:
        return 1
    print()
    print('  转换并与本机语料合并…')
    print('  注：本机 RecruitOps 导出的那部分只在这台机器上有，')
    print('      别人克隆后只有开源数据集那 9,218 条，数字会略有差异。')
    return run([PY, '-u', '-B', 'src/import_external_jds.py', '--merge'])


def step_ingest() -> int:
    hr('[4/5] 建库')
    if need_venv():
        return 1
    if DB.is_file() and DB.stat().st_size > 20 * 1048576:
        print('  data/rag.db 已存在（%.0f MB），跳过。' % (DB.stat().st_size / 1048576))
        print('  想重建就加 --force。')
        return 0
    if not CORPUS.is_file():
        print('  [错误] 没有 %s，先跑 python setup.py data' % CORPUS)
        return 1
    print('  切块 + 向量化，2 万多块，约 25 分钟。可以放着。')
    rc = run([PY, '-u', '-B', 'src/ingest.py', '--input', str(CORPUS),
              '--db', str(DB), '--reset',
              '--embed', 'fastembed', '--model', 'BAAI/bge-small-zh-v1.5',
              '--batch-size', '64'])
    if rc == 0:
        print()
        print('  给没有方向标签的岗位补 label（评测标注需要全库一致）…')
        run([PY, '-u', '-B', 'src/label_documents.py', '--db', str(DB)])
    return rc


def step_serve(port: int = 8000, open_browser: bool = True) -> int:
    hr('启动服务')
    if need_venv():
        return 1
    if not DB.is_file():
        print('  [错误] 还没有数据库。先跑：python setup.py all')
        return 1

    rc = run([PY, '-c',
              'import fastapi, uvicorn, fastembed, openpyxl'],
             quiet=True, check=False)
    if rc != 0:
        print('  [!] 依赖不全，先跑：python setup.py venv')
        return 1

    con = None
    try:
        import sqlite3
        con = sqlite3.connect('file:%s?mode=ro' % str(DB).replace('\\', '/'), uri=True)
        docs = con.execute('select count(*) from documents').fetchone()[0]
        chunks = con.execute('select count(*) from chunks').fetchone()[0]
        vecs = con.execute('select count(*) from chunks where embedding is not null').fetchone()[0]
        print('  库内：%d 岗位 / %d 切块 / %d 有向量' % (docs, chunks, vecs))
    except Exception as e:                                        # noqa: BLE001
        print('  [!] 读库失败：%s' % e)
    finally:
        if con is not None:
            con.close()

    url = 'http://127.0.0.1:%d' % port
    print()
    print('  演示页  : %s' % url)
    print('  接口文档: %s/docs' % url)
    print()
    print('  首次提问要加载 embedding 模型（约 20 秒），之后就快了。')
    has_key = bool(os.environ.get('DEEPSEEK_API_KEY') or _win_user_env('DEEPSEEK_API_KEY'))
    if has_key:
        print('  DEEPSEEK_API_KEY 已找到，/ask（LLM 生成答案）可用。')
    else:
        print('  提示：/ask 需要 DEEPSEEK_API_KEY，/compare 和演示页不需要。')
    print()
    print('  按 Ctrl+C 停止。')
    print()

    if open_browser:
        try:
            import threading
            def _open():
                time.sleep(6)
                import webbrowser
                webbrowser.open(url)
            threading.Thread(target=_open, daemon=True).start()
        except Exception:                                          # noqa: BLE001
            pass

    e = child_env()
    e['RAG_PORT'] = str(port)
    return subprocess.call([str(PY), '-u', '-B', 'src/api.py'], env=e, cwd=str(ROOT))


def step_check() -> int:
    hr('环境自检')
    # 只用 ASCII 标记：[OK] / [--]。
    # 中文 Windows 控制台默认是 GBK，✅❌ 这类字符会直接 UnicodeEncodeError。
    ok = lambda b: '[OK]' if b else '[--]'              # noqa: E731

    print('  Python (本次运行) : %s' % sys.version.split()[0])
    print('  虚拟环境          : %s %s' % (ok(PY.is_file()), PY if PY.is_file() else '(未创建)'))

    if PY.is_file():
        rc = run([PY, '-c', 'import fastapi, uvicorn, fastembed, openpyxl, jieba, numpy'],
                 quiet=True, check=False)
        print('  依赖              : %s' % ok(rc == 0))
        if rc != 0:
            print('                      （跑 python setup.py venv 安装）')

    print('  开源数据集 xlsx   : %s %s'
          % (ok(XLSX.is_file()),
             '(%.1f MB)' % (XLSX.stat().st_size / 1048576) if XLSX.is_file() else ''))
    print('  合并语料          : %s %s'
          % (ok(CORPUS.is_file()),
             '(%.1f MB)' % (CORPUS.stat().st_size / 1048576) if CORPUS.is_file() else ''))
    print('  数据库            : %s %s'
          % (ok(DB.is_file()),
             '(%.0f MB)' % (DB.stat().st_size / 1048576) if DB.is_file() else ''))

    if DB.is_file():
        try:
            import sqlite3
            con = sqlite3.connect('file:%s?mode=ro' % str(DB).replace('\\', '/'), uri=True)
            docs = con.execute('select count(*) from documents').fetchone()[0]
            chunks = con.execute('select count(*) from chunks').fetchone()[0]
            fts = con.execute('select count(*) from chunks_fts').fetchone()[0]
            vecs = con.execute('select count(*) from chunks where embedding is not null').fetchone()[0]
            uniq = con.execute('select count(distinct doc_key) from documents').fetchone()[0]
            con.close()
            print('    岗位 %d / 切块 %d / 向量 %d' % (docs, chunks, vecs))
            print('    FTS 与切块一致    : %s (%d vs %d)' % (ok(chunks == fts), chunks, fts))
            print('    doc_key 唯一      : %s (%d vs %d)' % (ok(docs == uniq), docs, uniq))
        except Exception as e:                                     # noqa: BLE001
            print('    [!] 读库失败：%s' % e)

    has_key = bool(os.environ.get('DEEPSEEK_API_KEY')
                   or _win_user_env('DEEPSEEK_API_KEY'))
    print('  DEEPSEEK_API_KEY  : %s'
          % ('[OK] 已设置' if has_key
             else '[--] 未设置（/ask 不可用，其余不受影响）'))
    print()
    print('  下一步：')
    if not PY.is_file():
        print('    python setup.py venv')
    elif not DB.is_file():
        print('    python setup.py all     # 建库（约 25 分钟）')
    else:
        print('    python setup.py serve   # 启动，然后打开 http://127.0.0.1:8000')
    return 0


# ─────────────── 入口 ───────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description='RAG 岗位问答系统 —— 一键环境准备 / 建库 / 启动',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('子命令')[1] if '子命令' in __doc__ else '')
    ap.add_argument('cmd', nargs='?', default='check',
                    choices=['check', 'venv', 'models', 'data', 'ingest',
                             'serve', 'all'])
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--force', action='store_true', help='建库时忽略已有 rag.db')
    args = ap.parse_args()

    print('=' * 72)
    print('RAG 岗位智能问答系统 —— setup')
    print('=' * 72)
    print('  项目目录: %s' % ROOT)

    if args.cmd == 'check':
        return step_check()
    if args.cmd == 'venv':
        return step_venv()
    if args.cmd == 'models':
        return step_models()
    if args.cmd == 'data':
        return step_data()
    if args.cmd == 'ingest':
        if args.force and DB.is_file():
            print('  --force：删除已有数据库重建')
            DB.unlink()
        return step_ingest()
    if args.cmd == 'serve':
        return step_serve(args.port, not args.no_browser)
    if args.cmd == 'all':
        for step in (step_venv, step_models, step_data, step_ingest):
            if step() != 0:
                print()
                print('  [中断] 上面这步失败了，修好再重跑 `python setup.py all`（已完成的步骤会跳过）。')
                return 1
        print()
        print('  全部完成。现在可以：python setup.py serve')
        return 0
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n  已中断。')
        sys.exit(130)
