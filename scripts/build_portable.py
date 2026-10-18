"""Publisher-side Windows x64 bundler. End users only extract and launch.

Run on the build machine with Python, pip and httpx available. Downloaded
artifacts are checked before extraction; no partial bundle is published.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import httpx

ROOT=Path(__file__).resolve().parents[1]
PYTHON_URL='https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg'
PYTHON_CATALOG='https://api.nuget.org/v3/catalog0/data/2025.04.08.14.01.15/python.3.12.10.json'
ENGINE_TAG='b11669'
ENGINE_URL=f'https://github.com/ggml-org/llama.cpp/releases/download/{ENGINE_TAG}/llama-{ENGINE_TAG}-bin-win-cpu-x64.zip'
# Published on the official GitHub release's expanded assets page.
ENGINE_SHA256='37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b'
MODEL_REPO='Qwen/Qwen2.5-0.5B-Instruct-GGUF'
MODEL_FILE='qwen2.5-0.5b-instruct-q4_k_m.gguf'


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.digest()


def download(client,url,target,expected=None,algorithm='sha256'):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and expected and digest(target,algorithm).hex()==expected:
        return
    temporary=target.with_suffix(target.suffix+'.part')
    try:
        with client.stream('GET',url) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for chunk in response.iter_bytes(1024*1024):
                    f.write(chunk)
        if expected and digest(temporary,algorithm).hex()!=expected:
            raise ValueError(f'Artifact checksum mismatch for {target.name}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def extract(archive,destination,prefix=''):
    destination=Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not item.filename.startswith(prefix) or item.is_dir():
                continue
            relative=item.filename[len(prefix):]
            if not relative:
                continue
            # Windows treats backslashes as separators; refuse these even on Linux.
            if '\\' in relative or ':' in relative or (item.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive member')
            target=(destination/relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive member escapes destination')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as source,target.open('wb') as sink:
                shutil.copyfileobj(source,sink)


def build(args):
    stage=Path(args.stage).resolve()
    cache=stage.parent/'portable-build-cache'
    cache.mkdir(parents=True,exist_ok=True)
    stage.mkdir(parents=True,exist_ok=True)
    if (stage/'data').exists():
        raise ValueError('Build directory contains user data; use a clean staging directory')
    manifest={'complete':False,'platform':'windows-x64','model_alias':'atlas-local','model_path':f'models/{MODEL_FILE}',
              'sources':{},'assets':{}}
    manifest_path=stage/'bundle.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for name in ['launcher.py','Start Atlas.cmd','READ ME.txt']:
        shutil.copy2(ROOT/'portable'/name,stage/name)
    shutil.copytree(ROOT/'personal_ai',stage/'app'/'personal_ai',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (stage/'licenses').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'LICENSE',stage/'licenses'/'workbench-MIT.txt')
    try:
        with httpx.Client(follow_redirects=True,timeout=120) as client:
            print('Preparing verified portable Python runtime…',flush=True)
            catalog=client.get(PYTHON_CATALOG)
            catalog.raise_for_status()
            catalog=catalog.json()
            if catalog.get('packageHashAlgorithm')!='SHA512' or catalog.get('authors')!='Python Software Foundation':
                raise ValueError('Unexpected Python publisher or hash metadata')
            expected=base64.b64decode(catalog['packageHash']).hex()
            runtime=cache/'python.3.12.10.nupkg'
            download(client,PYTHON_URL,runtime,expected,'sha512')
            extract(runtime,stage/'runtime',prefix='tools/')
            if not (stage/'runtime'/'python.exe').is_file():
                raise ValueError('Python package has no Windows runtime')
            manifest['sources']['python']={'url':PYTHON_URL,'catalog':PYTHON_CATALOG,'sha512':expected}

            print('Preparing Windows dependency wheels…',flush=True)
            wheels=cache/'wheels'
            wheels.mkdir(exist_ok=True)
            subprocess.run([sys.executable,'-m','pip','download','--only-binary=:all:','--platform','win_amd64','--python-version','312','--implementation','cp','--abi','cp312','--no-deps','-r',str(ROOT/'requirements.lock'),'colorama==0.4.6','-d',str(wheels)],check=True)
            # Pip verifies the package-index link hashes while downloading.
            for wheel in sorted(wheels.glob('*.whl')):
                extract(wheel,stage/'vendor')
            manifest['sources']['wheels']={p.name:digest(p).hex() for p in wheels.glob('*.whl')}

            print('Preparing verified local inference engine…',flush=True)
            engine=cache/f'llama-{ENGINE_TAG}-bin-win-cpu-x64.zip'
            download(client,ENGINE_URL,engine,ENGINE_SHA256)
            extract(engine,stage/'engine')
            servers=list((stage/'engine').rglob('llama-server.exe'))
            if len(servers)!=1:
                raise ValueError('Expected exactly one inference server')
            if servers[0].parent!=stage/'engine':
                for item in list(servers[0].parent.iterdir()):
                    if item.is_file():
                        shutil.copy2(item,stage/'engine'/item.name)
            download(client,f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{ENGINE_TAG}/LICENSE',stage/'licenses'/'llama.cpp-MIT.txt')
            manifest['sources']['engine']={'url':ENGINE_URL,'sha256':ENGINE_SHA256}

            print('Preparing licensed pretrained model (required for a complete package)…',flush=True)
            metadata=client.get(f'https://huggingface.co/api/models/{MODEL_REPO}',params={'blobs':'true'})
            metadata.raise_for_status()
            metadata=metadata.json()
            revision=metadata['sha']
            entry=next(f for f in metadata['siblings'] if f['rfilename']==MODEL_FILE)
            model_hash=entry['lfs']['sha256']
            if len(model_hash)!=64 or metadata.get('cardData',{}).get('license')!='apache-2.0':
                raise ValueError('Unexpected model digest or license; manual publisher review required')
            url=f'https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{MODEL_FILE}'
            download(client,url,stage/manifest['model_path'],model_hash)
            download(client,f'https://huggingface.co/{MODEL_REPO}/resolve/{revision}/LICENSE',stage/'licenses'/'Qwen-Apache-2.0.txt')
            manifest['sources']['model']={'url':url,'revision':revision,'sha256':model_hash,'license':'Apache-2.0'}
        manifest['assets']={str(p.relative_to(stage)).replace('\\','/'):digest(p).hex() for p in sorted(stage.rglob('*')) if p.is_file() and p!=manifest_path}
        manifest['complete']=True
        manifest_path.write_text(json.dumps(manifest,indent=2))
        output=Path(args.output).resolve()
        output.parent.mkdir(parents=True,exist_ok=True)
        # Build to a temporary ZIP; publish only after every required asset passed.
        temporary=output.with_suffix('.zip.part')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    z.write(path,'Atlas/'+str(path.relative_to(stage)))
        temporary.replace(output)
        print('Complete portable ZIP created:',output)
    except BaseException as exc:
        manifest['complete']=False
        manifest['build_error']=str(exc)
        manifest_path.write_text(json.dumps(manifest,indent=2))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='/workspace/deliverables/atlas-portable-stage')
    parser.add_argument('--output',default='/workspace/deliverables/atlas-windows-portable.zip')
    build(parser.parse_args())
