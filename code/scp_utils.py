# -*- coding: utf-8 -*-
# @Time    : 2023/05/25 19:22
# @Author  : zhangchong
# @Site    :
# @File    : scp_util.py
# @Software: Code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import logging
import asyncio
import asyncssh
import tqdm
from fire import Fire

sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
logger = logging.getLogger(__name__)

def pretty_file_size(size_bytes):
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def is_local_to_remote(direction):
    return direction in ("local_to_remote", "l2r")

async def transfer_file(
    src_path, dest_path, conn, sftp, chunk_size, direction, progress_bar=True
):
    # 获取源文件和目标文件的大小
    if is_local_to_remote(direction):
        src_size_bytes = os.path.getsize(src_path)
        try:
            dest_size = await conn.run(f"stat -c %s {dest_path}", check=True)
            dest_size_bytes = int(dest_size.stdout.strip())
            print(sftp.getsize(dest_size))
        except asyncssh.ProcessError:
            dest_size_bytes = 0
    else:
        dest_size_bytes = os.path.getsize(dest_path) 
        remote_file_size = await conn.run(f"stat -c %s {src_path}", check=True)
        src_size_bytes = int(remote_file_size.stdout.strip())
        print(sftp.getsize(src_path))

    print(f'Source file: {src_path}')
    print(f'Source file size: {pretty_file_size(src_size_bytes)}')
    print(f'Destination file: {dest_path}')
    print(f'Destination exists size: {pretty_file_size(dest_size_bytes)}')

    # 如果目标文件大小与源文件大小相同，则无需传输
    if dest_size_bytes == src_size_bytes:
        print(f"{dest_path} already exists.")
        return

    # 断点续传：从已存在的目标文件大小的位置开始传输
    if is_local_to_remote(direction):
        with open(src_path, "rb") as src_file:
            src_file.seek(dest_size_bytes)

            async with sftp.open(dest_path, "ab") as dest_file:
                if progress_bar:
                    progress = tqdm.tqdm(
                        total=src_size_bytes,
                        initial=dest_size_bytes,
                        unit="B",
                        unit_scale=True,
                        desc=src_path,
                    )

                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break

                    await dest_file.write(chunk)

                    if progress_bar:
                        progress.update(len(chunk))

                if progress_bar:
                    progress.close()
    else:
        async with sftp.open(src_path, "rb") as src_file:
            src_file.seek(dest_size_bytes)

            with open(dest_path, "ab") as dest_file:
                if progress_bar:
                    progress = tqdm.tqdm(
                        total=src_size_bytes,
                        initial=dest_size_bytes,
                        unit="B",
                        unit_scale=True,
                        desc=src_path,
                    )

                while True:
                    chunk = await src_file.read(chunk_size)
                    if not chunk:
                        break

                    dest_file.write(chunk)

                    if progress_bar:
                        progress.update(len(chunk))

                if progress_bar:
                    progress.close()


async def copy_directory(src_path, dest_path, conn, sftp,chunk_size, direction):
    # Copy file from local to remote via ssh 
    if is_local_to_remote(direction):
        await conn.run(f"mkdir -p {dest_path}", check=True)
        for item in os.listdir(src_path):
            local_item = os.path.join(src_path, item)
            remote_item = os.path.join(dest_path, item)

            if os.path.isfile(local_item):
                await transfer_file(local_item, remote_item, conn, sftp, chunk_size, direction)
            elif os.path.isdir(local_item):
                await copy_directory(local_item, remote_item, conn, sftp, chunk_size, direction)
    else:
        os.makedirs(dest_path, exist_ok=True)
        async for entry in sftp.listdir(src_path):
            remote_item = os.path.join(src_path, entry.filename)
            local_item = os.path.join(dest_path, entry.filename)

            if entry.longname.startswith("-"):  # It's a file
                await transfer_file(remote_item, local_item, conn, sftp, chunk_size, direction)
            elif entry.longname.startswith("d"):  # It's a directory
                await copy_directory(remote_item, local_item, conn, sftp, chunk_size, direction) 
        

async def async_scp(src_path, dest_path, host, port, username, password, chunk_size, direction):
    async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
        async with conn.start_sftp_client() as sftp:
            if is_local_to_remote(direction): 
                if os.path.isfile(src_path):
                    await transfer_file(src_path, dest_path, conn, sftp, chunk_size, direction)
                elif os.path.isdir(src_path):
                    await copy_directory(src_path, dest_path, conn, sftp, chunk_size, direction)
            else:
                if await sftp.isfile(src_path):
                    await transfer_file(src_path, dest_path, conn, sftp, chunk_size, direction)
                elif await sftp.isdir(src_path):
                    await copy_directory(src_path, dest_path, conn, sftp, chunk_size, direction)






async def scp_main(
    local_path="path/to/local/file",
    remote_path="path/to/remote/file",
    host="example.com",
    port=22,
    username="your_username",
    password="your_password",
    chunk_size=1048576,
    direction="r2l"
):
    if direction not in ("local_to_remote", "remote_to_local", "l2r", "r2l"):
        raise ValueError("Invalid direction, must be 'local_to_remote' or 'remote_to_local' or 'l2r' or 'r2l'")
    await async_scp(local_path, remote_path, host, port, username, password, chunk_size, direction)


def main(
**kwargs
):
    print(f'Scp task: \nFrom:{kwargs["local_path"]}\nTo=>:{kwargs["remote_path"]}\nHost:{kwargs["host"]}\nUsername:{kwargs["username"]}\nPassword:{kwargs["password"]}\n')
    print('Start scp task')
    asyncio.run(scp_main(**kwargs))


if __name__ == "__main__":
    Fire(main)
