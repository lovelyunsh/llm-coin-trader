"""API key encryption/decryption using Fernet (cryptography library)."""

import base64
import hashlib
import json
from importlib import import_module
from pathlib import Path


class KeyManager:
    """Manages encrypted API keys with version support."""

    VERSION = 1

    @staticmethod
    def _get_fernet() -> type:
        try:
            return import_module("cryptography.fernet").Fernet
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "cryptography is required for KeyManager. Install with `pip install cryptography`."
            ) from e

    @staticmethod
    def _derive_fernet_key(master_key: str) -> bytes:
        key_bytes = hashlib.sha256(master_key.encode()).digest()
        return base64.urlsafe_b64encode(key_bytes)

    @staticmethod
    def encrypt_keys(
        api_key: str, api_secret: str, key_file: Path, master_key: str
    ) -> None:
        fernet_key = KeyManager._derive_fernet_key(master_key)
        cipher = KeyManager._get_fernet()(fernet_key)
        payload = {
            "version": KeyManager.VERSION,
            "api_key": api_key,
            "api_secret": api_secret,
        }
        encrypted = cipher.encrypt(json.dumps(payload).encode())
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(encrypted)

    @staticmethod
    def decrypt_keys(key_file: Path, master_key: str) -> tuple[str, str]:
        if not key_file.exists():
            raise FileNotFoundError(f"Encrypted key file not found: {key_file}")
        fernet_key = KeyManager._derive_fernet_key(master_key)
        cipher = KeyManager._get_fernet()(fernet_key)
        encrypted = key_file.read_bytes()
        try:
            plaintext = cipher.decrypt(encrypted)
            payload = json.loads(plaintext.decode())
        except Exception as e:
            raise ValueError(f"Failed to decrypt keys: {e}") from e
        if payload.get("version") != KeyManager.VERSION:
            raise ValueError(
                f"Key version mismatch: expected {KeyManager.VERSION}, got {payload.get('version')}"
            )
        return payload["api_key"], payload["api_secret"]
