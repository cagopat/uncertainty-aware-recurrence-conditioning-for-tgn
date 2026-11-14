"""SW1 inherited integrity and immutable storage helpers. Standard library only."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
import zipfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, f'Duplicate JSON key: {k}')
        out[k] = v
    return out


def loads(text):
    return json.loads(text, object_pairs_hook=_unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read_json(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        require(not path.is_symlink() and read_json(path) == value, f'Immutable JSON differs: {path}')
    else:
        write_json(path, value)


def safe_relative(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name, 'Invalid relative path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.parts and '..' not in p.parts and ':' not in p.parts[0], f'Unsafe path: {name}')
    require(p.as_posix() == name.rstrip('/'), f'Noncanonical path: {name}')
    return p


def safe_file(root, name):
    root = Path(root)
    require(not root.is_symlink(), f'Symlink root: {root}')
    p = root
    for part in safe_relative(name).parts:
        p = p / part
        require(not p.is_symlink(), f'Symlink forbidden: {p}')
    require(p.is_file(), f'Missing file: {p}')
    return p


def copy_exact(source, target, expected=None):
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink(), f'Invalid source: {source}')
    expected = sha(source) if expected is None else expected
    require(sha(source) == expected, f'Source checksum mismatch: {source}')
    require(not target.is_symlink(), f'Symlink target: {target}')
    for parent in target.parents:
        require(not parent.is_symlink(), f'Symlink parent: {parent}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(target.is_file() and sha(target) == expected, f'Existing file differs: {target}')
    else:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        require(sha(target) == expected, f'Copy read-back failed: {target}')
    return target


def inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), f'Invalid directory: {root}')
    out = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), f'Symlink in inventory: {p}')
        require(p.is_dir() or p.is_file(), f'Special file in inventory: {p}')
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha(p)
    return out


def verify_package(root, expected_manifest=None):
    root = Path(root)
    manifest_path = safe_file(root, 'PACKAGE_MANIFEST.json')
    if expected_manifest is not None:
        require(sha(manifest_path) == expected_manifest, 'Package manifest differs from notebook pin')
    manifest = read_json(manifest_path)
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Empty package manifest')
    for name, digest in manifest['files'].items():
        require(sha(safe_file(root, name)) == digest, f'Package changed: {name}')
    return manifest


def stage_package(source, destination, expected_manifest):
    """Authenticate a folder or one package ZIP and copy only manifest payload."""
    source, destination = Path(source), Path(destination)
    require(source.exists() and not source.is_symlink(), f'Missing package: {source}')
    if source.is_dir():
        root = source
    else:
        require(source.suffix.lower() == '.zip', 'Package must be an expanded folder or ZIP')
        scratch = Path(tempfile.mkdtemp(prefix='sw1_package_')).resolve()
        with zipfile.ZipFile(source) as z:
            members = safe_zip_members(z)
            manifests = [i.filename for i in members if i.filename.endswith('/PACKAGE_MANIFEST.json') or i.filename == 'PACKAGE_MANIFEST.json']
            require(len(manifests) == 1, f'Expected exactly one package manifest, found {manifests}')
            manifest_member = manifests[0]
            prefix = manifest_member[:-len('PACKAGE_MANIFEST.json')]
            raw = z.read(manifest_member)
            require(hashlib.sha256(raw).hexdigest() == expected_manifest, 'ZIP package manifest differs from notebook pin')
            manifest = loads(raw.decode('utf-8'))
            names = {'PACKAGE_MANIFEST.json': hashlib.sha256(raw).hexdigest(), **manifest['files']}
            for name, expected in names.items():
                safe_relative(name)
                member = prefix + name
                require(member in z.namelist(), f'Missing package member: {member}')
                target = scratch / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                require(sha(target) == expected, f'ZIP member changed: {name}')
        root = scratch
    manifest = verify_package(root, expected_manifest)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['files'].items():
        copy_exact(safe_file(root, name), destination / name, expected)
    copy_exact(root / 'PACKAGE_MANIFEST.json', destination / 'PACKAGE_MANIFEST.json', expected_manifest)
    verify_package(destination, expected_manifest)
    return destination


def safe_zip_members(archive):
    members = archive.infolist()
    require(len(members) <= 150000, 'Too many ZIP members')
    require(len({i.filename for i in members}) == len(members), 'Duplicate ZIP members')
    require(sum(i.file_size for i in members) <= 20 * 1024**3, 'ZIP exceeds 20 GiB uncompressed limit')
    for i in members:
        safe_relative(i.filename)
        mode = i.external_attr >> 16
        require(not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), f'Special ZIP member: {i.filename}')
        require(not i.flag_bits & 1, 'Encrypted ZIP unsupported')
    require(archive.testzip() is None, 'ZIP CRC failure')
    return members


def seal_unit(payload, unit_root, identity, inputs, attempt):
    payload, unit_root = Path(payload), Path(unit_root)
    require((payload / 'REPORT.json').is_file(), 'Worker did not emit REPORT.json')
    require(not (unit_root / 'COMPLETE.json').exists(), f'Unit already sealed: {unit_root}')
    hashes = inventory(payload)
    publication = unit_root / 'publications' / attempt / 'payload'
    for name, digest in hashes.items():
        copy_exact(safe_file(payload, name), publication / name, digest)
    require(inventory(publication) == hashes, 'Publication inventory differs')
    receipt = {'identity': identity, 'inputs': inputs, 'files': hashes,
               'payload': publication.relative_to(unit_root).as_posix(), 'attempt': attempt, 'completed_at': stamp()}
    write_json(publication.parent / 'RECEIPT.json', receipt)
    write_json(unit_root / 'COMPLETE.json', receipt)
    return receipt


def verify_unit(unit_root, identity):
    unit_root = Path(unit_root)
    receipt = read_json(safe_file(unit_root, 'COMPLETE.json'))
    require(receipt['identity'] == identity, f'Unit identity differs: {unit_root}')
    relative = safe_relative(receipt['payload'])
    payload = unit_root.joinpath(*relative.parts)
    require(read_json(safe_file(payload.parent, 'RECEIPT.json')) == receipt, 'Publication receipt differs')
    require(inventory(payload) == receipt['files'], f'Unit payload changed: {unit_root}')
    require('REPORT.json' in receipt['files'], 'Sealed unit lacks report')
    return receipt, payload


def restore_unit(unit_root, destination, identity, expected_inputs=None):
    unit_root, destination = Path(unit_root), Path(destination)
    if not (unit_root / 'COMPLETE.json').exists():
        return False
    receipt, payload = verify_unit(unit_root, identity)
    if expected_inputs is not None:
        require(receipt['inputs'] == expected_inputs, f'Unit input identities differ: {unit_root}')
    for name, digest in receipt['files'].items():
        copy_exact(safe_file(payload, name), destination / name, digest)
    require(inventory(destination) == receipt['files'], f'Local restored output has extra or changed files: {destination}')
    return receipt


def archive_folder(folder, destination):
    folder, destination = Path(folder), Path(destination)
    files = inventory(folder)
    require('EXPORT_MANIFEST.json' not in files, 'Reserved archive manifest filename')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in files:
            z.write(folder / name, name)
        z.writestr('EXPORT_MANIFEST.json', json.dumps({'files': files}, sort_keys=True, indent=2))
    verify_export_archive(destination)
    return files


def verify_export_archive(path):
    with zipfile.ZipFile(path) as z:
        members = safe_zip_members(z)
        names = {i.filename for i in members if not i.is_dir()}
        manifest = loads(z.read('EXPORT_MANIFEST.json').decode('utf-8'))['files']
        require(names == set(manifest) | {'EXPORT_MANIFEST.json'}, 'Export inventory mismatch')
        for name, expected in manifest.items():
            h = hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(b)
            require(h.hexdigest() == expected, f'Export member changed: {name}')
    return manifest


def verify_export_receipt(archive):
    archive = Path(archive)
    receipt = read_json(Path(str(archive) + '.sha256.json'))
    require(receipt['sha256'] == sha(archive) and receipt['bytes'] == archive.stat().st_size, 'Export outer checksum/size mismatch')
    files = verify_export_archive(archive)
    require(receipt['files'] == files, 'Export outer/inner inventories disagree')
    return receipt


@contextmanager
def session_lock(remote, old_runtime_stopped=False):
    remote = Path(remote)
    remote.mkdir(parents=True, exist_ok=True)
    lock = remote / 'SESSION_RUNNING.json'
    if lock.exists():
        require(old_runtime_stopped, 'Session lock exists. Confirm the earlier runtime is stopped before requesting lock recovery.')
        old = read_json(lock)
        copy_exact(lock, remote / 'lock_history' / (stamp() + '_' + uuid.uuid4().hex + '.json'))
        require(read_json(lock) == old, 'Lock changed during recovery')
        lock.unlink()
    token = uuid.uuid4().hex
    write_json(lock, {'session': token, 'created_at': stamp(), 'pid': os.getpid()})
    try:
        yield token
    finally:
        if lock.exists() and read_json(lock).get('session') == token:
            copy_exact(lock, remote / 'lock_history' / (stamp() + '_' + token + '.released.json'))
            lock.unlink()
