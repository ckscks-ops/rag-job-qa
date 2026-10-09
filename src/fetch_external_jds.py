# -*- coding: utf-8 -*-
"""
下载开源中文招聘 JD 数据集（补充数据分析方向的语料）。

数据源：https://github.com/Rayair019/Job-posting-data
    10,114 条 2025 年中文招聘 JD，来自 Boss直聘 / 智联 / 猎聘，
    分五类：data analysis / software and algorithm / biomedical /
            economics and finance / large language model algorithm

⚠️ 为什么走镜像
    github.com 与 codeload.github.com 在本机**连不上**（TLS 握手超时），
    但 api.github.com 正常、gh-proxy.com / ghproxy.net 正常。
    所以：元信息走 api.github.com，文件下载走镜像。

⚠️ 许可
    该仓库**没有 LICENSE**。数据来自公开招聘平台的岗位描述（本身是公开信息），
    用于个人求职作品集可以接受，但**不要商用、不要再分发原始文件**。
    引用时注明来源。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

REPO = 'Rayair019/Job-posting-data'
FILE = 'job_posting_data.xlsx'
BRANCH = 'main'
OUT = Path(__file__).resolve().parent.parent / 'data' / 'external' / FILE

MIRRORS = [
    'https://gh-proxy.com/https://raw.githubusercontent.com/%s/%s/%s' % (REPO, BRANCH, FILE),
    'https://ghproxy.net/https://github.com/%s/raw/%s/%s' % (REPO, BRANCH, FILE),
    'https://ghproxy.net/https://raw.githubusercontent.com/%s/%s/%s' % (REPO, BRANCH, FILE),
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size > 1_000_000:
        print('[跳过] 已存在 %s (%.2f MB)' % (OUT, OUT.stat().st_size / 1024 / 1024))
        return 0

    headers = {'User-Agent': 'Mozilla/5.0 (job-rag-research)'}
    for url in MIRRORS:
        host = url.split('/')[2]
        print('尝试镜像: %s' % host)
        t0 = time.perf_counter()
        try:
            with requests.get(url, headers=headers, stream=True, timeout=90) as r:
                if r.status_code != 200:
                    print('   HTTP %d，换下一个' % r.status_code)
                    continue
                total = int(r.headers.get('Content-Length') or 0)
                got = 0
                tmp = OUT.with_suffix('.part')
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(262144):
                        f.write(chunk)
                        got += len(chunk)
                        if total and got % (1024 * 1024) < 262144:
                            print('   %.1f / %.1f MB' % (got / 1048576, total / 1048576))
                dt = time.perf_counter() - t0
                if total and got < total:
                    print('   不完整（%d / %d 字节），换下一个' % (got, total))
                    tmp.unlink(missing_ok=True)
                    continue
                tmp.replace(OUT)
                print('   ✅ 下载完成 %.2f MB，用时 %.1f 秒（%.0f KB/s）'
                      % (got / 1048576, dt, got / 1024 / dt))
                return 0
        except Exception as e:                                # noqa: BLE001
            print('   失败: %s: %s' % (type(e).__name__, str(e)[:120]))

    print('\n所有镜像都失败。可选做法：')
    print('  1. 在浏览器里打开 https://github.com/%s 手动下载' % REPO)
    print('  2. 用别的代理')
    return 1


if __name__ == '__main__':
    sys.exit(main())
