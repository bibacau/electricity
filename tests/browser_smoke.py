"""Optional real-browser check; requires Playwright and an installed Chromium.

python tests/browser_smoke.py --python .venv/bin/python --browser /usr/bin/chromium
"""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright, expect

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--python',required=True)
parser.add_argument('--browser')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0))
    port=sock.getsockname()[1]
with tempfile.TemporaryDirectory(prefix='personal-ai-browser-') as directory:
    process=subprocess.Popen([str(Path(args.python).resolve()),'-m','personal_ai','--data',directory,'--port',str(port)],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        base=f'http://127.0.0.1:{port}'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(100):
            try:
                with opener.open(base+'/api/health',timeout=1) as response:
                    assert response.status==200
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError('Application exited before readiness')
                time.sleep(0.1)
        else:
            raise RuntimeError('Application did not become ready')
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path=args.browser,args=['--no-sandbox'] if os.name!='nt' else [])
            page=browser.new_page(viewport={'width':1440,'height':1000})
            page.set_default_timeout(10000)
            errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base)
            expect(page.locator('#ai-name')).to_have_text('Atlas')
            page.locator('[data-page=memory]').click()
            page.locator('#memory-text').fill('Browser smoke test: concise explanations')
            page.locator('#memory-verified').check()
            page.get_by_role('button',name='Save memory',exact=True).click()
            page.get_by_text('Browser smoke test: concise explanations',exact=True).wait_for()
            page.get_by_role('button',name='Edit',exact=True).click()
            page.locator('#memory-text').fill('Browser smoke test: edited memory')
            page.get_by_role('button',name='Save memory',exact=True).click()
            page.get_by_text('Browser smoke test: edited memory',exact=True).wait_for()
            page.reload()
            page.locator('[data-page=memory]').click()
            page.get_by_text('Browser smoke test: edited memory',exact=True).wait_for()
            page.locator('[data-page=personality]').click()
            page.locator('input[name=name]').fill('Atlas Browser Test')
            page.get_by_role('button',name='Save preferences').click()
            page.get_by_text('Preferences saved. New conversations use these settings.',exact=True).wait_for()
            page.locator('[data-page=learning]').click()
            page.locator('#episodes').fill('100')
            page.locator('#grid-size').fill('3')
            page.get_by_role('button',name='Start experiment',exact=True).click()
            page.get_by_text('completed',exact=True).wait_for(timeout=15000)
            page.locator('[data-page=chat]').click()
            # An empty selected model fails honestly even if an unrelated Ollama instance is running.
            page.locator('#chat-input').fill('Hello local model')
            page.get_by_role('button',name='Send message',exact=True).click()
            page.get_by_text('Request failed:',exact=False).wait_for()
            page.screenshot(path=str(Path(tempfile.gettempdir())/'personal-ai-desktop.png'),full_page=True)
            assert not errors,errors
            page.set_viewport_size({'width':390,'height':844})
            page.reload()
            expect(page.locator('#ai-name')).to_have_text('Atlas Browser Test')
            assert page.evaluate('() => document.documentElement.scrollWidth <= window.innerWidth'),'Horizontal overflow'
            page.screenshot(path=str(Path(tempfile.gettempdir())/'personal-ai-mobile.png'),full_page=True)
            browser.close()
            print('Browser checks passed: memory CRUD/reload, personality, real grid job, missing-model error, mobile layout; no page errors.')
    finally:
        process.terminate()
        process.wait(timeout=15)
