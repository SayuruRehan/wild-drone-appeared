"""Exercise the notebook's real Git clone against a local fake-auth repository."""

import base64
import getpass
import json
import os
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def test_hidden_clone_token_is_not_persisted_by_inherited_git_helper(tmp_path, monkeypatch):
    fake_token = 'FAKE_TEST_TOKEN'
    http_root = tmp_path/'http'
    http_root.mkdir()
    seed = tmp_path/'seed'
    subprocess.run(['git', 'init', '-q', str(seed)], check=True)
    subprocess.run(['git', '-C', str(seed), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '--allow-empty',
                    '-qm', 'Fixture'], check=True)
    bare = http_root/'repo.git'
    subprocess.run(['git', 'clone', '--bare', '-q', str(seed), str(bare)], check=True)
    subprocess.run(['git', '--git-dir', str(bare), 'update-server-info'], check=True)
    expected_auth = 'Basic '+base64.b64encode(f'x-access-token:{fake_token}'.encode()).decode()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Authorization') != expected_auth:
                self.send_response(401)
                self.send_header('WWW-Authenticate', 'Basic realm="test"')
                self.end_headers()
                return
            self.path = self.path.split('?')[0]
            super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=http_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    credentials = tmp_path/'credentials'
    config = tmp_path/'gitconfig'
    config.write_text(f'[credential]\n\thelper = store --file={credentials}\n')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setattr(getpass, 'getpass', lambda _: fake_token)
    notebook_path = Path(__file__).parents[1]/'notebooks'/'colab_experiments.ipynb'
    notebook = json.loads(notebook_path.read_text())
    source = ''.join(notebook['cells'][2]['source'])
    source = source.replace('/content/wild-drone-appeared', str(tmp_path/'cloned'))
    source = source.replace('https://github.com/SayuruRehan/wild-drone-appeared.git',
                            f'http://127.0.0.1:{server.server_port}/repo.git')
    before = Path.cwd()
    namespace = {}
    try:
        exec(compile(source, 'colab-clone-cell', 'exec'), namespace)
    finally:
        os.chdir(before)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not credentials.exists() or fake_token not in credentials.read_text()
    assert namespace['token'] == ''
    assert 'WILD_DRONE_GITHUB_TOKEN' not in namespace['child_env']
