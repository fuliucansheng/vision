"""Windows wheel relocation guards; no CUDA device or private fixture required."""
import importlib.util
import csv
import io
import sys
import zipfile
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
if sys.platform != 'win32':
    raise unittest.SkipTest('Windows wheel relocation tests')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vision_relocate', ROOT / 'packaging/wheel/relocate.py')
relocate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relocate)

class RelocationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lib = self.root / 'wheel/torchvision'
        self.lib.mkdir(parents=True)
        self.image = self.pe(self.lib / 'image_stable.pyd', 43620)

    def pe(self, path, machine):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray(64)
        data[:2] = b'MZ'
        data[60:64] = 64 .to_bytes(4, 'little')
        path.write_bytes(data + b'PE\x00\x00' + machine.to_bytes(2, 'little'))
        return path

    def cuda_env(self):
        return mock.patch.dict(os.environ, {'CUDA_PATH': str(self.root / 'cuda')}, clear=True)

    def test_arm64_native_directory(self):
        dll = self.pe(self.root / 'cuda/bin/arm64/nvjpeg64_13.dll', 43620)
        with self.cuda_env():
            self.assertEqual(relocate.find_cuda_dll(dll.name, str(self.image)), str(dll))

    def test_legacy_bin_directory(self):
        dll = self.pe(self.root / 'cuda/bin/nvjpeg64_13.dll', 43620)
        with self.cuda_env():
            self.assertEqual(relocate.find_cuda_dll(dll.name, str(self.image)), str(dll))

    def test_wrong_architecture_is_rejected(self):
        self.pe(self.root / 'cuda/bin/arm64/nvjpeg64_13.dll', 34404)
        with self.cuda_env():
            self.assertIsNone(relocate.find_cuda_dll('nvjpeg64_13.dll', str(self.image)))

    def test_invalid_pe_is_rejected(self):
        p = self.root / 'invalid.dll'
        p.write_bytes(b'not a PE')
        self.assertIsNone(relocate.pe_machine(str(p)))

    def deps(self, _dumpbin, binary):
        return ['nvjpeg64_13.dll'] if Path(binary).name == 'image_stable.pyd' else []

    def invoke(self):
        with mock.patch.object(relocate, 'find_dll_dependencies', side_effect=self.deps), mock.patch.object(relocate, 'find_program', return_value=None):
            relocate.relocate_dll_library('dumpbin', str(self.lib.parent), str(self.lib), self.image.name)

    def test_missing_nvjpeg_stops_packaging(self):
        with self.cuda_env(), self.assertRaisesRegex(FileNotFoundError, 'Required nvjpeg64_13.dll'):
            self.invoke()

    def test_existing_bundled_library_is_not_self_copied(self):
        dll = self.pe(self.lib / 'nvjpeg64_13.dll', 43620)
        before = dll.read_bytes()
        with self.cuda_env():
            self.invoke()
        self.assertEqual(dll.read_bytes(), before)

    def test_matching_cuda_library_is_bundled(self):
        dll = self.pe(self.root / 'cuda/bin/arm64/nvjpeg64_13.dll', 43620)
        with self.cuda_env():
            self.invoke()
        self.assertEqual((self.lib / dll.name).read_bytes(), dll.read_bytes())

    def test_x64_legacy_layout(self):
        image = self.pe(self.lib / 'x64.pyd', 34404)
        dll = self.pe(self.root / 'cuda/bin/nvjpeg64_13.dll', 34404)
        with self.cuda_env():
            self.assertEqual(relocate.find_cuda_dll(dll.name, str(image)), str(dll))

    def test_wheel_record_names_match_zip_members(self):
        output = self.root / 'record-wheel'
        info = output / 'demo-1.0.dist-info'
        info.mkdir(parents=True)
        (output / 'package').mkdir()
        (output / 'package/module.py').write_text('value = 1\n', encoding='utf-8')
        (info / 'METADATA').write_text('Name: demo\nVersion: 1.0\n', encoding='utf-8')
        wheel = self.root / 'demo-1.0-py3-none-any.whl'
        wheel.write_bytes(b'')
        relocate.compress_wheel(str(output), str(wheel), str(self.root), 'demo-1.0-py3-none-any')
        with zipfile.ZipFile(wheel) as z:
            record = list(csv.reader(io.StringIO(z.read('demo-1.0.dist-info/RECORD').decode('utf-8'))))
            self.assertEqual({r[0] for r in record}, {n for n in z.namelist() if not n.endswith('/')})
            self.assertTrue(all(('\\' not in r[0] for r in record)))
if __name__ == '__main__':
    unittest.main(verbosity=2)
